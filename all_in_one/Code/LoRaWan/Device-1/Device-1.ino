#include <badger.h>
bool LoRa_add_sensitivity(uint8_t sensitivity);
// enter your own key
const uint8_t devEUI[8] = { 0x00, 0x04, 0xA3, 0x0B, 0x00, 0x1E, 0x66, 0xD4 };

const uint8_t appKey[16] = { 0x8E, 0x8A, 0x1B, 0x34, 0x41, 0xFB, 0xD1, 0x62, 0xA2, 0x0C, 0x3F, 0x16, 0x0F, 0xA8, 0xD1, 0xD3 };

const uint8_t appEUI[8] = { 0x70, 0xB3, 0xD5, 0x7E, 0xD0, 0x00, 0xA2, 0x22 };

String payload = "";
bool new_message = false;

int id = 15;
int total_place = 7;
int total_free = 5;
byte data_payload[6];

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
    Serial.println(payload);
    if (payload.equals("Send")) {
      Serial.println("Message is correct");
      data_payload[0] = highByte(id);
      data_payload[1] = lowByte(id);
      data_payload[2] = highByte(total_place);
      data_payload[3] = lowByte(total_place);
      data_payload[4] = highByte(total_free);
      data_payload[5] = lowByte(total_free);
      SendPayload(data_payload);
      Serial.println("Sending");
      
    }
    Serial.println("Refresh the payload");
    payload = "";
    memset(data_payload, 0, sizeof(data_payload));
    new_message = false;
  }
}

bool SendPayload (byte message[]) {
  bool success = false;
  int send_try = 0;

  while (send_try <= 9) {
    if (LoRa_send(1, (uint8_t*)message, sizeof(message), 1)) {
      success = true;
      send_try = 10;
      delay(10000);
    }
    send_try ++;
  }
  return success;
}
