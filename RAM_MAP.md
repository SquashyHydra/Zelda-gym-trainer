# The Legend of Zelda: Link's Awakening (Game Boy)/RAM map

| RAM | Purpose |
|-----|---------|
| D401 | Destination data byte 1: 00 - overworld, 01 - dungeon, 02 - side view area |
| D402 | Destination data byte 2: Values from 00 to 1F accepted. FF is Color Dungeon |
| D403 | Destination data byte 3: Room number. Must appear on map or it will lead to an empty room |
| D404 - D405 | Destination data X and Y co-ordinates |
| D700 - D79B | Currently loaded map |
| D800 - D8FF | World map status |

Each screen status is represented by a byte, which is a combination of the following masks :

| Mask | Meaning |
|------|---------|
| 00 | Unexplored |
| 10 | Changed from initial status (for example sword taken on the beach or dungeon opened with key) |
| 20 | Owl talked |
| 80 | Visited |

For example, visiting the first dungeon's screen (80) and opening it with the key (10) would put that byte at 90

| RAM | Purpose |
|-----|---------|
| DB00 - DB01 | Your currently held items. |
| DB02 - DB0B | Inventory |
| DB0C | Flippers (01=have) |
| DB0D | Potion (01=have) |
| DB0E | Current item in trading game (01=Yoshi, 0E=magnifier) |
| DB0F | Number of secret shells |
| DB10-DB14 | Dungeons entrance keys (01=have) |
| DB15 | Number of golden leaves |
| DB16 - DB3D | Beginning of dungeon item flags. 5 bytes fo each dungeon, 5th byte is quantity of keys for that dungeon |
| DB43 | Power bracelet level |
| DB44 | Shield level |
| DB45 | Number of arrows |
| DB49 | Ocarina songs in possession (3 bits mask, 0=no songs, 7=all songs) |
| DB4A | Ocarina selected song |
| DB4C | Magic powder quantity |
| DB4D | Number of bombs |
| DB4E | Sword level |
| DB56-DB58 | Number of times the character died for each save slot (one byte per save slot) |
| DB5A | Current health. Each increment of 08h is one full heart, each increment of 04h is one-half heart. |
| DB5B | Maximum health. Simply counts the number of hearts Link has in hex. Max recommended value is 0Eh (14 hearts). |
| DB5D-DB5E | Number of rupees (for 999 put 0999) |
| DB65-DB6C | Instruments for every dungeon, 00=no instrument, 03=have instrument |
| DB76 | Max magic powder |
| DB77 | Max bombs |
| DB78 | Max arrows |
| DBAE | Your position on the 8x8 dungeon grid |
| DBD0 | Quantity of keys in posession |


| Id | Item |
|----|------|
| 01 | Sword |
| 02 | Bombs |
| 03 | Power bracelet |
| 04 | Shield |
| 05 | Bow |
| 06 | Hookshot |
| 07 | Fire rod |
| 08 | Pegasus boots |
| 09 | Ocarina |
| 0A | Feather |
| 0B | Shovel |
| 0C | Magic powder |
| 0D | Boomrang |