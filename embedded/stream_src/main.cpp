#include <Arduino.h>
#include <WebServer.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "img_converters.h"
#include "wifi_credentials.h"

namespace {
WebServer server(80);

camera_config_t camera_config() {
  camera_config_t config{};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sccb_sda = 26;
  config.pin_sccb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QQVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_DRAM;
  return config;
}

void handle_root() {
  server.send(
      200, "text/html",
      "<!doctype html><html><head><title>ESP32-CAM diagnostic</title>"
      "<meta name='viewport' content='width=device-width,initial-scale=1'>"
      "</head><body><h1>ESP32-CAM live diagnostic</h1>"
      "<p>MJPEG stream with continuous LED color-policy classification.</p>"
      "<p id='status'>Analyzing live stream...</p>"
      "<p id='details'></p>"
      "<img id='camera' src='/stream' style='max-width:100%;image-rendering:auto'>"
      "<canvas id='analysis' width='160' height='120' hidden></canvas>"
      "<script>"
      "const image=document.getElementById('camera');"
      "const canvas=document.getElementById('analysis');"
      "const context=canvas.getContext('2d',{willReadFrequently:true});"
      "const status=document.getElementById('status');"
      "const details=document.getElementById('details');"
      "function analyze(){"
      " if(!image.naturalWidth){return;}"
      " context.drawImage(image,0,0,160,120);"
      " const pixels=context.getImageData(0,0,160,120).data;"
      " let blue=0,green=0,yellow=0,visible=0;"
      " for(let i=0;i<pixels.length;i+=4){"
      "  const r=pixels[i],g=pixels[i+1],b=pixels[i+2];"
      "  if(r+g+b>90)visible++;"
      "  if(b>80 && b>r*1.2 && b>g*1.1 && b-r>20)blue++;"
      "  if(g>80 && g>r*1.15 && g>b*1.15 && g-r>20)green++;"
      "  if(r>80 && g>80 && r>b*1.2 && g>b*1.2 && Math.abs(r-g)<90)yellow++;"
      " }"
      " const denominator=visible||1;"
      " const ratios={blue:blue/denominator,green:green/denominator,yellow:yellow/denominator};"
      " let label='UNKNOWN / INSUFFICIENT COLOR SIGNAL';"
      " let color='black';"
      " if(ratios.blue>=0.01 && ratios.blue>=ratios.green && ratios.blue>=ratios.yellow){"
      "  label='BLUE LOCATE INDICATOR - HEALTH NOT CLASSIFIED'; color='blue';"
      " } else if(ratios.green>=0.01 && ratios.green>=ratios.yellow){"
      "  label='SERVER STATUS: WORKING / GREEN'; color='green';"
      " } else if(ratios.yellow>=0.01){"
      "  label='SERVER STATUS: FAULT / YELLOW'; color='darkgoldenrod';"
      " }"
      " status.textContent=label;"
      " status.style.color=color;"
      " details.textContent='Visible pixels: '+visible+' | green: '+(ratios.green*100).toFixed(2)+'% | yellow: '+(ratios.yellow*100).toFixed(2)+'% | blue: '+(ratios.blue*100).toFixed(2)+'%';"
      "}"
      "setInterval(analyze,500);"
      "</script></body></html>");
}

void handle_stream() {
  WiFiClient client = server.client();
  client.print("HTTP/1.1 200 OK\r\n"
              "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
              "Cache-Control: no-cache\r\n"
              "Connection: close\r\n\r\n");
  while (client.connected()) {
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame == nullptr) {
      break;
    }
    uint8_t* jpeg_buffer = nullptr;
    size_t jpeg_length = 0;
    const bool converted = frame2jpg(
        frame, 80, &jpeg_buffer, &jpeg_length);
    if (!converted) {
      esp_camera_fb_return(frame);
      break;
    }
    client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                  jpeg_length);
    client.write(jpeg_buffer, jpeg_length);
    client.print("\r\n");
    free(jpeg_buffer);
    esp_camera_fb_return(frame);
    delay(100);
  }
  client.stop();
}
}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\nstandalone_camera_stream_start");

  camera_config_t config = camera_config();
  const esp_err_t camera_status = esp_camera_init(&config);
  Serial.printf("camera_init=%s\npixel_format=RGB565\nsoftware_jpeg=true\n",
                esp_err_to_name(camera_status));
  if (camera_status != ESP_OK) {
    return;
  }
  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor != nullptr) {
    sensor->set_brightness(sensor, 2);
    sensor->set_gain_ctrl(sensor, 1);
    sensor->set_exposure_ctrl(sensor, 1);
    sensor->set_awb_gain(sensor, 1);
  }

  WiFi.mode(WIFI_STA);
  const int network_count = WiFi.scanNetworks();
  Serial.printf("wifi_scan_count=%d\n", network_count);
  for (int index = 0; index < network_count; ++index) {
    Serial.printf("wifi_scan_%d=%s,rssi=%d,channel=%d\n", index,
                  WiFi.SSID(index).c_str(), WiFi.RSSI(index),
                  WiFi.channel(index));
  }
  WiFi.scanDelete();
  WiFi.begin(kWifiSsid, kWifiPassword);
  Serial.printf("wifi_connecting=ssid_%s\n", kWifiSsid);
  const uint32_t wifi_deadline = millis() + 20000;
  while (WiFi.status() != WL_CONNECTED && millis() < wifi_deadline) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("wifi_connect=failed_status_%d\n", WiFi.status());
    return;
  }
  Serial.printf("wifi_connect=ok\nssid=%s\nip=%s\nurl=http://%s/\n",
                kWifiSsid, WiFi.localIP().toString().c_str(),
                WiFi.localIP().toString().c_str());
  server.on("/", HTTP_GET, handle_root);
  server.on("/stream", HTTP_GET, handle_stream);
  server.begin();
  Serial.println("http_server=ready");
}

void loop() {
  server.handleClient();
}
