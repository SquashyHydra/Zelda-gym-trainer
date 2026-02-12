from pyboy import PyBoy
from pyboy.utils import WindowEvent
from sys import path
from os import system, makedirs, listdir
from os.path import join

import pyboy.logging

state_dir = f"{path[0]}\\States"

def ram(addr):
    return env.memory[addr]

try:
    open_state = input("Load game state [y/n]: ")

    with PyBoy(f'{path[0]}\\ROM\\Zelda.gb') as env:
        if open_state.lower() == "y":
            for file in listdir(state_dir):
                print(file.replace(".state", ""))
            state = input("Select a state: ")
            with open(join(state_dir, f"{state}.state"), "rb") as f:
                env.load_state(f)
        system('cls')
        while env.tick():
            system('cls')
            env.send_input(WindowEvent.PRESS_BUTTON_START)
            env.send_input(WindowEvent.RELEASE_BUTTON_START)
            print(f"Items: {ram(0xDB0C)}")
            print('Deaths ', ram(0xDB57))
            print('Health ', f'{ram(0xDB5A) / 8}', '/', f'{ram(0xDB5B)}.0')
            print(f"Co-ords x: {ram(0xD404)} y: {ram(0xD405)}")
            print(f"[World]\n{ram(0xD401)}, {ram(0xD402)}, {ram(0xD403)}")
            print(f"[Map Loaded]\n{0xD700}, {0xD701}, {0xD702}, {0xD703}, {0xD704}, {0xD705}, {0xD706}, {0xD707}, {0xD708}, {0xD709}\n{0xD70A}, {0xD71A}, {0xD72A}, {0xD73A}, {0xD74A}, {0xD75A}, {0xD76A}, {0xD77A}, {0xD78A}, {0xD79A}\n{0xD70B}, {0xD71B}, {0xD72B}, {0xD73B}, {0xD74B}, {0xD75B}, {0xD76B}, {0xD77B}, {0xD78B}, {0xD79B}")
            print(f"Inventory")
            print(f"{0xDB02}, {0xDB03}, {0xDB04}")
            print(f"{0xDB05}, {0xDB06}, {0xDB07}")
            print(f"{0xDB09}, {0xDB0A}, {0xDB0B}")
            print(f"World Map]")
            print(f"{ram(0xD80A)}, {ram(0xD81A)}, {ram(0xD82A)}, {ram(0xD83A)}, {ram(0xD84A)}, {ram(0xD85A)}, {ram(0xD86A)}, {ram(0xD87A)}, {ram(0xD88A)}, {ram(0xD89A)}")
            print(f"{ram(0xD80B)}, {ram(0xD81B)}, {ram(0xD82B)}, {ram(0xD83B)}, {ram(0xD84B)}, {ram(0xD85B)}, {ram(0xD86B)}, {ram(0xD87B)}, {ram(0xD88B)}, {ram(0xD89B)}")
            print(f"{ram(0xD80C)}, {ram(0xD81C)}, {ram(0xD82C)}, {ram(0xD83C)}, {ram(0xD84C)}, {ram(0xD85C)}, {ram(0xD86C)}, {ram(0xD87C)}, {ram(0xD88C)}, {ram(0xD89C)}")
            print(f"{ram(0xD80D)}, {ram(0xD81D)}, {ram(0xD82D)}, {ram(0xD83D)}, {ram(0xD84D)}, {ram(0xD85D)}, {ram(0xD86D)}, {ram(0xD87D)}, {ram(0xD88D)}, {ram(0xD89D)}")
            print(f"{ram(0xD80E)}, {ram(0xD81E)}, {ram(0xD82E)}, {ram(0xD83E)}, {ram(0xD84E)}, {ram(0xD85E)}, {ram(0xD86E)}, {ram(0xD87E)}, {ram(0xD88E)}, {ram(0xD89E)}")
            print(f"{ram(0xD80F)}, {ram(0xD81F)}, {ram(0xD82F)}, {ram(0xD83F)}, {ram(0xD84F)}, {ram(0xD85F)}, {ram(0xD86F)}, {ram(0xD87F)}, {ram(0xD88F)}, {ram(0xD89F)}") # 60 Tiles
            print("\n")
            print(f"{ram(0xD800)}, {ram(0xD801)}, {ram(0xD802)}, {ram(0xD803)}, {ram(0xD804)}, {ram(0xD805)}, {ram(0xD806)}, {ram(0xD807)}, {ram(0xD808)}, {ram(0xD809)}")
            print(f"{ram(0xD810)}, {ram(0xD811)}, {ram(0xD812)}, {ram(0xD813)}, {ram(0xD814)}, {ram(0xD815)}, {ram(0xD816)}, {ram(0xD817)}, {ram(0xD818)}, {ram(0xD819)}")
            print(f"{ram(0xD820)}, {ram(0xD821)}, {ram(0xD822)}, {ram(0xD823)}, {ram(0xD824)}, {ram(0xD825)}, {ram(0xD826)}, {ram(0xD827)}, {ram(0xD828)}, {ram(0xD829)}")
            print(f"{ram(0xD830)}, {ram(0xD831)}, {ram(0xD832)}, {ram(0xD833)}, {ram(0xD834)}, {ram(0xD835)}, {ram(0xD836)}, {ram(0xD837)}, {ram(0xD838)}, {ram(0xD839)}")
            print(f"{ram(0xD840)}, {ram(0xD841)}, {ram(0xD842)}, {ram(0xD843)}, {ram(0xD844)}, {ram(0xD845)}, {ram(0xD846)}, {ram(0xD847)}, {ram(0xD848)}, {ram(0xD849)}")
            print(f"{ram(0xD850)}, {ram(0xD851)}, {ram(0xD852)}, {ram(0xD853)}, {ram(0xD854)}, {ram(0xD855)}, {ram(0xD856)}, {ram(0xD857)}, {ram(0xD858)}, {ram(0xD859)}")
            print(f"{ram(0xD860)}, {ram(0xD861)}, {ram(0xD862)}, {ram(0xD863)}, {ram(0xD864)}, {ram(0xD865)}, {ram(0xD866)}, {ram(0xD867)}, {ram(0xD868)}, {ram(0xD869)}")
            print(f"{ram(0xD870)}, {ram(0xD871)}, {ram(0xD872)}, {ram(0xD873)}, {ram(0xD874)}, {ram(0xD875)}, {ram(0xD876)}, {ram(0xD877)}, {ram(0xD878)}, {ram(0xD879)}")
            print(f"{ram(0xD880)}, {ram(0xD881)}, {ram(0xD882)}, {ram(0xD883)}, {ram(0xD884)}, {ram(0xD885)}, {ram(0xD886)}, {ram(0xD887)}, {ram(0xD888)}, {ram(0xD889)}")
            print(f"{ram(0xD890)}, {ram(0xD891)}, {ram(0xD892)}, {ram(0xD893)}, {ram(0xD894)}, {ram(0xD895)}, {ram(0xD896)}, {ram(0xD897)}, {ram(0xD898)}, {ram(0xD899)}") # 100 Tiles
            print("\n")
            print(f"{ram(0xD8A0)}, {ram(0xD8A1)}, {ram(0xD8A2)}, {ram(0xD8A3)}, {ram(0xD8A4)}, {ram(0xD8A5)}, {ram(0xD8A6)}, {ram(0xD8A7)}, {ram(0xD8A8)}, {ram(0xD8A9)}")
            print(f"{ram(0xD8B0)}, {ram(0xD8B1)}, {ram(0xD8B2)}, {ram(0xD8B3)}, {ram(0xD8B4)}, {ram(0xD8B5)}, {ram(0xD8B6)}, {ram(0xD8B7)}, {ram(0xD8B8)}, {ram(0xD8B9)}")
            print(f"{ram(0xD8C0)}, {ram(0xD8C1)}, {ram(0xD8C2)}, {ram(0xD8C3)}, {ram(0xD8C4)}, {ram(0xD8C5)}, {ram(0xD8C6)}, {ram(0xD8C7)}, {ram(0xD8C8)}, {ram(0xD8C9)}")
            print(f"{ram(0xD8D0)}, {ram(0xD8D1)}, {ram(0xD8D2)}, {ram(0xD8D3)}, {ram(0xD8D4)}, {ram(0xD8D5)}, {ram(0xD8D6)}, {ram(0xD8D7)}, {ram(0xD8D8)}, {ram(0xD8D9)}")
            print(f"{ram(0xD8E0)}, {ram(0xD8E1)}, {ram(0xD8E2)}, {ram(0xD8E3)}, {ram(0xD8E4)}, {ram(0xD8E5)}, {ram(0xD8E6)}, {ram(0xD8E7)}, {ram(0xD8E8)}, {ram(0xD8E9)}")
            print(f"{ram(0xD8F0)}, {ram(0xD8F1)}, {ram(0xD8F2)}, {ram(0xD8F3)}, {ram(0xD8F4)}, {ram(0xD8F5)}, {ram(0xD8F6)}, {ram(0xD8F7)}, {ram(0xD8F8)}, {ram(0xD8F9)}") # 60 Tiles
            print("\n")
            print(f"{ram(0xD8AA)}, {ram(0xD8BA)}, {ram(0xD8CA)}, {ram(0xD8DA)}, {ram(0xD8EA)}, {ram(0xD8FA)}")
            print(f"{ram(0xD8AB)}, {ram(0xD8BB)}, {ram(0xD8CB)}, {ram(0xD8DB)}, {ram(0xD8EB)}, {ram(0xD8FB)}")
            print(f"{ram(0xD8AC)}, {ram(0xD8BC)}, {ram(0xD8CC)}, {ram(0xD8DC)}, {ram(0xD8EC)}, {ram(0xD8FC)}")
            print(f"{ram(0xD8AD)}, {ram(0xD8BD)}, {ram(0xD8CD)}, {ram(0xD8DD)}, {ram(0xD8ED)}, {ram(0xD8FD)}")
            print(f"{ram(0xD8AE)}, {ram(0xD8BE)}, {ram(0xD8CE)}, {ram(0xD8DE)}, {ram(0xD8EE)}, {ram(0xD8FE)}")
            print(f"{ram(0xD8AF)}, {ram(0xD8BF)}, {ram(0xD8CF)}, {ram(0xD8DF)}, {ram(0xD8EF)}, {ram(0xD8FF)}") # 36 Tiles
            pass
finally:
    state_name = input("name the state: ")
    makedirs(state_dir, exist_ok=True)
    with open(join(state_dir, f"{state_name}.state"), "wb") as f:
        env.save_state(f)

    env.stop()