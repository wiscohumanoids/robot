/*
 * Wireless E-Stop transmitter (ESP32-S3, ESP-IDF).
 *
 * Joins the RK3588's Wi-Fi hotspot and sends UDP packets to the receiver:
 *   "ALIVE <seq>"  every 50 ms while the button is NOT pressed
 *   "STOP <seq>"   every 20 ms while the button IS pressed
 *
 * The receiver cuts power on STOP *or* when ALIVE stops arriving for 150 ms,
 * so anything that silences this board (dead battery, crash, out of range,
 * Wi-Fi drop) also stops the robot. This code never has to "decide" to stop
 * on a fault; it only has to keep proving that it is alive and the button is
 * released.
 *
 * Button wiring: normally-closed (NC) contact between the GPIO and GND, with
 * the internal pull-up enabled.
 *   not pressed -> contact closed -> pin reads 0
 *   pressed     -> contact open   -> pin reads 1 (pull-up)
 *   wire cut    -> contact open   -> pin reads 1 -> treated as pressed
 *
 * Settings (SSID, receiver IP, GPIO, periods): `idf.py menuconfig` ->
 * "Wireless E-Stop Transmitter", or edit main/Kconfig.projbuild defaults.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"

static const char *TAG = "estop_tx";

#define BUTTON_GPIO     ((gpio_num_t)CONFIG_ESTOP_BUTTON_GPIO)
#define LOOP_MS         5   /* how often the button is sampled */

static volatile bool s_wifi_connected = false;

/* ---------------- Wi-Fi ---------------- */

/* Called by the Wi-Fi driver on connect / disconnect / got-IP events.
 * On any disconnect we just try again forever. While disconnected, nothing
 * reaches the receiver, so the robot is already stopped - that is the
 * intended fail-safe, not an error to hide. */
static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_wifi_connected) {
            ESP_LOGW(TAG, "Wi-Fi lost -> receiver will time out and cut power. Reconnecting...");
        }
        s_wifi_connected = false;
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "Wi-Fi connected, my IP " IPSTR, IP2STR(&ev->ip_info.ip));
        s_wifi_connected = true;
    }
}

static void wifi_start(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL));

    wifi_config_t cfg = {0};
    strncpy((char *)cfg.sta.ssid, CONFIG_ESTOP_WIFI_SSID, sizeof(cfg.sta.ssid) - 1);
    strncpy((char *)cfg.sta.password, CONFIG_ESTOP_WIFI_PASSWORD, sizeof(cfg.sta.password) - 1);

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    /* Power save lets the radio sleep between beacons, adding up to ~100 ms
     * of latency. Turn it off: we care about latency, not battery here. */
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_LOGI(TAG, "Connecting to SSID \"%s\"...", CONFIG_ESTOP_WIFI_SSID);
}

/* ---------------- Button ---------------- */

static void button_init(void)
{
    gpio_config_t io = {
        .pin_bit_mask = 1ULL << BUTTON_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,   /* ~45 kOhm internal; add an external 10k for long wires */
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io));
}

/* 1 = pressed OR wire broken (NC contact open). */
static bool button_raw_pressed(void)
{
    return gpio_get_level(BUTTON_GPIO) == 1;
}

/* ---------------- Main loop ---------------- */

static void transmit_task(void *arg)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "socket() failed; rebooting");
        esp_restart();   /* no packets would be sent anyway; reboot is the clean retry */
    }

    struct sockaddr_in dest = {0};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(CONFIG_ESTOP_RECEIVER_PORT);
    dest.sin_addr.s_addr = inet_addr(CONFIG_ESTOP_RECEIVER_IP);

    uint32_t seq = 0;
    bool pressed = true;            /* start as "pressed" until the button proves released */
    int64_t released_since_us = -1; /* when the raw reading last became "released" */
    int64_t next_send_us = 0;
    char msg[32];

    ESP_LOGI(TAG, "Sending to %s:%d (ALIVE every %d ms, STOP every %d ms)",
             CONFIG_ESTOP_RECEIVER_IP, CONFIG_ESTOP_RECEIVER_PORT,
             CONFIG_ESTOP_HEARTBEAT_MS, CONFIG_ESTOP_STOP_PERIOD_MS);

    while (true) {
        int64_t now = esp_timer_get_time();

        /* Asymmetric debounce: a press counts immediately (fast stop);
         * a release must hold for ESTOP_RELEASE_MS (contact bounce or a
         * flickering broken wire cannot briefly look "released"). */
        if (button_raw_pressed()) {
            released_since_us = -1;
            if (!pressed) {
                pressed = true;
                ESP_LOGW(TAG, "E-STOP PRESSED -> sending STOP");
                next_send_us = now;   /* send STOP right now, don't wait for the next slot */
            }
        } else {
            if (released_since_us < 0) {
                released_since_us = now;
            }
            if (pressed && now - released_since_us >= (int64_t)CONFIG_ESTOP_RELEASE_MS * 1000) {
                pressed = false;
                ESP_LOGI(TAG, "E-stop released -> sending ALIVE");
                next_send_us = now;
            }
        }

        if (now >= next_send_us) {
            if (s_wifi_connected) {
                seq++;
                int len = snprintf(msg, sizeof(msg), "%s %lu", pressed ? "STOP" : "ALIVE",
                                   (unsigned long)seq);
                /* UDP: fire and forget. A lost packet is fine - the next one
                 * comes 20-50 ms later, and silence itself means "stop". */
                sendto(sock, msg, len, 0, (struct sockaddr *)&dest, sizeof(dest));
            }
            int period_ms = pressed ? CONFIG_ESTOP_STOP_PERIOD_MS : CONFIG_ESTOP_HEARTBEAT_MS;
            next_send_us = now + (int64_t)period_ms * 1000;
        }

        vTaskDelay(pdMS_TO_TICKS(LOOP_MS));
    }
}

void app_main(void)
{
    /* NVS is required by the Wi-Fi driver (it stores calibration data there). */
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    button_init();
    wifi_start();
    xTaskCreate(transmit_task, "estop_tx", 4096, NULL, 10, NULL);
}
