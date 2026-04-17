#include <Adafruit_GFX.h>    // Core graphics library
#include <Adafruit_ST7789.h> // Hardware-specific library

// The Adafruit board profile automatically defines TFT_CS, TFT_DC, and TFT_RST
Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);

void setup() {
  Serial.begin(115200);

  // 1. Turn on the physical power to the screen
  pinMode(TFT_I2C_POWER, OUTPUT);
  digitalWrite(TFT_I2C_POWER, HIGH);
  delay(10); 

  // 2. Turn on the backlight so you can actually see it
  pinMode(TFT_BACKLITE, OUTPUT);
  digitalWrite(TFT_BACKLITE, HIGH);

  // 3. Initialize the screen (the Feather has a 135x240 pixel display)
  tft.init(135, 240); 
  
  // 4. Set up the display orientation and background
  tft.setRotation(1);               // 1 = Landscape mode
  tft.fillScreen(ST77XX_BLACK);     // Clear the screen with a black background
  
  // 5. Write the text!
  tft.setCursor(30, 50);            // Start writing at X:30, Y:50
  tft.setTextColor(ST77XX_CYAN);    // Make the text Cyan
  tft.setTextSize(3);               // Set the font size
  
  tft.print("Hello World!");        // Print the message to the screen
}

void loop() {
  // A static "Hello World" doesn't need anything in the loop
}