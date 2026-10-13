#include <Arduino.h>
#include <driver/mcpwm.h>
#include <esp_mac.h>
#include <esp_timer.h>

#if __has_include("wifi_secrets.h")
#include "wifi_secrets.h"
#endif
#ifndef ENABLE_WIFI_UDP
#define ENABLE_WIFI_UDP 0
#endif

#if ENABLE_WIFI_UDP
#include <WiFi.h>
#include <WiFiUdp.h>
#ifndef WIFI_SSID
#define WIFI_SSID ""
#endif
#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD ""
#endif
#ifndef UDP_PORT
#define UDP_PORT 4210
#endif
#endif

#ifndef SA100_GPIO
#define SA100_GPIO 4
#endif
#ifndef SAMPLE_RATE_HZ
#define SAMPLE_RATE_HZ 200
#endif

static constexpr uint32_t PROTOCOL_VERSION = 1;
static constexpr size_t CAPTURE_QUEUE_LENGTH = 128;

struct PulseSample {
  uint32_t high_ticks;
  uint32_t period_ticks;
  int64_t t_device_us;
  bool valid_edges;
};

static QueueHandle_t capture_queue;
struct TxPacket { size_t length; char data[320]; };
static QueueHandle_t tx_queue;
static SemaphoreHandle_t serial_mutex;
static volatile uint32_t capture_drops = 0;
static uint32_t tx_drops = 0;
static char boot_id[17];
static uint32_t sequence_number = 0;

#if ENABLE_WIFI_UDP
static WiFiUDP udp;
static IPAddress udp_peer;
static uint16_t udp_peer_port = 0;
static SemaphoreHandle_t udp_mutex;
#endif

static bool IRAM_ATTR on_capture(mcpwm_unit_t, mcpwm_capture_channel_id_t, const cap_event_data_t *event, void *) {
  static uint32_t last_rise = 0;
  static uint32_t fall_after_rise = 0;
  static bool have_rise = false;
  static bool have_fall = false;
  BaseType_t wake = pdFALSE;
  if (event->cap_edge == MCPWM_POS_EDGE) {
    if (have_rise) {
      PulseSample sample;
      // Unsigned subtraction deliberately handles the 32-bit capture counter wrap.
      sample.period_ticks = event->cap_value - last_rise;
      sample.high_ticks = fall_after_rise - last_rise;
      sample.t_device_us = esp_timer_get_time();
      sample.valid_edges = have_fall && sample.high_ticks > 0 && sample.high_ticks < sample.period_ticks;
      if (xQueueSendFromISR(capture_queue, &sample, &wake) != pdTRUE) {
        ++capture_drops;
      }
    }
    last_rise = event->cap_value;
    have_rise = true;
    have_fall = false;
  } else if (event->cap_edge == MCPWM_NEG_EDGE && have_rise && !have_fall) {
    fall_after_rise = event->cap_value;
    have_fall = true;
  }
  return wake == pdTRUE;
}

static void emit_line(const char *line) {
  size_t length = strlen(line);
  xSemaphoreTake(serial_mutex, portMAX_DELAY);
  if (Serial.write(reinterpret_cast<const uint8_t *>(line), length) != length) ++tx_drops;
  xSemaphoreGive(serial_mutex);
#if ENABLE_WIFI_UDP
  xSemaphoreTake(udp_mutex, portMAX_DELAY);
  if (WiFi.status() == WL_CONNECTED && udp_peer_port != 0) {
    if (udp.beginPacket(udp_peer, udp_peer_port)) {
      udp.write(reinterpret_cast<const uint8_t *>(line), length);
      if (!udp.endPacket()) ++tx_drops;
    } else {
      ++tx_drops;
    }
  }
  xSemaphoreGive(udp_mutex);
#endif
}

static void sender_task(void *) {
  TxPacket packet;
  for (;;) {
    if (xQueueReceive(tx_queue, &packet, portMAX_DELAY) == pdTRUE) emit_line(packet.data);
  }
}

static void handle_command(const char *line, bool via_udp = false, int64_t receive_us = -1) {
  if (strstr(line, "\"type\":\"sync_req\"") == nullptr) return;
  const char *id_key = strstr(line, "\"id\":");
  const char *t0_key = strstr(line, "\"t0_pc_ns\":");
  if (!id_key || !t0_key) return;
  unsigned long request_id = strtoul(id_key + 5, nullptr, 10);
  unsigned long long t0 = strtoull(t0_key + 11, nullptr, 10);
  int64_t t1 = receive_us >= 0 ? receive_us : esp_timer_get_time();
  char response[240];
  if (!via_udp) {
    xSemaphoreTake(serial_mutex, portMAX_DELAY);
    int64_t t2 = esp_timer_get_time();
    snprintf(response, sizeof(response), "{\"v\":%lu,\"type\":\"sync_resp\",\"id\":%lu,\"t0_pc_ns\":%llu,\"t1_device_us\":%lld,\"t2_device_us\":%lld,\"boot_id\":\"%s\"}\n", static_cast<unsigned long>(PROTOCOL_VERSION), request_id, t0, static_cast<long long>(t1), static_cast<long long>(t2), boot_id);
    Serial.write(response);
    xSemaphoreGive(serial_mutex);
  }
#if ENABLE_WIFI_UDP
  else if (udp_peer_port) {
    xSemaphoreTake(udp_mutex, portMAX_DELAY);
    int64_t t2 = esp_timer_get_time();
    snprintf(response, sizeof(response), "{\"v\":%lu,\"type\":\"sync_resp\",\"id\":%lu,\"t0_pc_ns\":%llu,\"t1_device_us\":%lld,\"t2_device_us\":%lld,\"boot_id\":\"%s\"}\n", static_cast<unsigned long>(PROTOCOL_VERSION), request_id, t0, static_cast<long long>(t1), static_cast<long long>(t2), boot_id);
    udp.beginPacket(udp_peer, udp_peer_port); udp.write(reinterpret_cast<uint8_t *>(response), strlen(response)); udp.endPacket();
    xSemaphoreGive(udp_mutex);
  }
#endif
}

static void poll_commands() {
  static char serial_line[256];
  static size_t serial_used = 0;
  while (Serial.available()) {
    char ch = static_cast<char>(Serial.read());
    if (ch == '\n') {
      serial_line[serial_used] = 0; handle_command(serial_line); serial_used = 0;
    } else if (serial_used + 1 < sizeof(serial_line)) {
      serial_line[serial_used++] = ch;
    } else {
      serial_used = 0;
    }
  }
#if ENABLE_WIFI_UDP
  xSemaphoreTake(udp_mutex, portMAX_DELAY);
  int packet_size = udp.parsePacket();
  if (packet_size > 0) {
    char buffer[256];
    int count = udp.read(buffer, sizeof(buffer) - 1); buffer[max(0, count)] = 0;
    udp_peer = udp.remoteIP(); udp_peer_port = udp.remotePort();
    int64_t receive_us = esp_timer_get_time();
    xSemaphoreGive(udp_mutex);
    handle_command(buffer, true, receive_us);
  } else {
    xSemaphoreGive(udp_mutex);
  }
#endif
}

void setup() {
  Serial.setTxBufferSize(2048);
  Serial.begin(921600);
  uint8_t mac[6]; esp_read_mac(mac, ESP_MAC_WIFI_STA);
  snprintf(boot_id, sizeof(boot_id), "%02X%02X%02X%02X%02X%02X%04X", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], static_cast<unsigned>(esp_random() & 0xffff));
  capture_queue = xQueueCreate(CAPTURE_QUEUE_LENGTH, sizeof(PulseSample));
  tx_queue = xQueueCreate(16, sizeof(TxPacket));
  serial_mutex = xSemaphoreCreateMutex();
#if ENABLE_WIFI_UDP
  udp_mutex = xSemaphoreCreateMutex();
#endif
  xTaskCreate(sender_task, "protocol_tx", 4096, nullptr, 1, nullptr);
  pinMode(SA100_GPIO, INPUT_PULLDOWN);
  ESP_ERROR_CHECK(mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM_CAP_0, SA100_GPIO));
  mcpwm_capture_config_t capture_config = {};
  capture_config.cap_edge = MCPWM_BOTH_EDGE;
  capture_config.cap_prescale = 1;
  capture_config.capture_cb = on_capture;
  ESP_ERROR_CHECK(mcpwm_capture_enable_channel(MCPWM_UNIT_0, MCPWM_SELECT_CAP0, &capture_config));

#if ENABLE_WIFI_UDP
  WiFi.mode(WIFI_STA); WiFi.begin(WIFI_SSID, WIFI_PASSWORD); udp.begin(UDP_PORT);
#endif
}

void loop() {
  poll_commands();
  static PulseSample newest = {};
  static bool have_sample = false;
  PulseSample incoming;
  while (xQueueReceive(capture_queue, &incoming, 0) == pdTRUE) {
    newest = incoming; have_sample = true;
  }
  static uint32_t last_send_ms = 0;
  const uint32_t interval_ms = max<uint32_t>(1, 1000 / SAMPLE_RATE_HZ);
  if (have_sample && millis() - last_send_ms >= interval_ms) {
    last_send_ms = millis();
    const double position = newest.period_ticks ? (static_cast<double>(newest.high_ticks) * 4098.0 / newest.period_ticks - 1.0) : -1.0;
    const bool valid = newest.valid_edges && newest.period_ticks > 1000 && position >= 0.0 && position <= 4095.0;
    // 4096 counts per revolution; unlike the vendor example this does not divide the angle by 4098.
    const double angle = valid ? position * 360.0 / 4096.0 : 0.0;
    TxPacket packet = {};
    int length = snprintf(packet.data, sizeof(packet.data), "{\"v\":1,\"type\":\"sample\",\"boot_id\":\"%s\",\"seq\":%lu,\"t_device_us\":%lld,\"high_ticks\":%lu,\"period_ticks\":%lu,\"raw_angle_deg\":%.8f,\"valid\":%s,\"capture_drops\":%lu,\"tx_drops\":%lu}\n",
      boot_id, static_cast<unsigned long>(sequence_number++), static_cast<long long>(newest.t_device_us),
      static_cast<unsigned long>(newest.high_ticks), static_cast<unsigned long>(newest.period_ticks), angle,
      valid ? "true" : "false", static_cast<unsigned long>(capture_drops), static_cast<unsigned long>(tx_drops));
    packet.length = length > 0 ? static_cast<size_t>(length) : 0;
    if (packet.length > 0 && packet.length < sizeof(packet.data) && xQueueSend(tx_queue, &packet, 0) == pdTRUE) {
      have_sample = false;
    } else {
      ++tx_drops;
      have_sample = false;
    }
  }
  delay(0);
}
