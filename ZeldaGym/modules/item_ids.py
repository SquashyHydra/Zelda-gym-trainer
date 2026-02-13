def holding_item(item1, item2):
    Items = {
            0: "Empty",
            1: "Sword",
            2: "Bombs",
            3: "Power bracelet",
            4: "Shield",
            5: "Bow",
            6: "Hookshot",
            7: "Fire rod",
            8: "Pegasus boots",
            9: "Ocarina",
            10: "Feather",
            11: "Shovel",
            12: "Magic powder",
            13: "Boomrang",
        }
    return (Items.get(int(item1), f"Unknown({item1})"), Items.get(int(item2), f"Unknown({item2})"))