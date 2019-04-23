#include <badger.h>
bool LoRa_add_sensitivity(uint8_t sensitivity);
// enter your own key
const uint8_t devEUI[8] = { 0x00, 0x04, 0xA3, 0x0B, 0x00, 0x1E, 0x66, 0xD4 };

const uint8_t appKey[16] = { 0x8E, 0x8A, 0x1B, 0x34, 0x41, 0xFB, 0xD1, 0x62, 0xA2, 0x0C, 0x3F, 0x16, 0x0F, 0xA8, 0xD1, 0xD3 };

const uint8_t appEUI[8] = { 0x70, 0xB3, 0xD5, 0x7E, 0xD0, 0x00, 0xA2, 0x22 };

String payload = "";
bool new_message = false;
String key = "";
char* id = 15;
char* total_place = 7;
char* total_free = 5;
uint8_t data_payload[4];

void setup() {
  // open a serial connection
  Serial.begin(9600);
  badger_init();
  LoRa_add_sensitivity(1);
  delay(3000);
  while (! LoRa_init(devEUI, appEUI, appKey, true)) {
  }
}

void loop() {
  while (Serial.available()) {
    delay(2);  //delay to allow byte to arrive in input buffer
    char c = Serial.read();
    payload += c;
    new_message = true;
  }
  if (new_message) {
    payload.trim();
    char str_payload[payload.length() + 1] ;
    payload.toCharArray(str_payload, payload.length() + 1);
    key = strtok(str_payload, "-");
    id = strtok(NULL, "-");
    total_place =  strtok(NULL, "-");
    total_free =  (strtok(NULL, "-"));
    //key = payload.substring(0, 3);
    //id = (uint8_t) payload.substring(4, 7).toInt();
    //total_place = (uint8_t) payload.substring(8, 11).toInt();
    //total_free = (uint8_t) payload.substring(12, 15).toInt();

    if (key.equals("New")) {
      //  Serial.println("Message is correct");
      data_payload[0] = (uint8_t)atoi(id);
      data_payload[1] = (uint8_t)atoi(total_place);
      data_payload[2] = (uint8_t)atoi(total_free);
      SendPayload(data_payload);
      // Serial.println("Sending");

    }
    //  Serial.println("Refresh the payload");
    payload = "";
    key = "";
    id = 0;
    total_place = 0;
    total_free = 0;
    memset(data_payload, 0, sizeof(data_payload));
    new_message = false;
  }
}

bool SendPayload (uint8_t message[]) {
  bool success = false;
  int send_try = 0;

  while (send_try <= 9) {
    if (LoRa_send(1, (uint8_t*)message, (sizeof(message) + 1), 1)) {
      success = true;
      send_try = 10;
      delay(10000);
    }
    send_try ++;
  }
  return success;
}
