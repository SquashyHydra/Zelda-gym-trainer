from pyboy.utils import WindowEvent

valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PASS,
            WindowEvent.PASS
        ]

release_arrows = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_UP,
        ]

release_buttons = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.PASS,
            WindowEvent.PASS
        ]

def get_action_name(action):
    actions = {
        0: "Arrow Down",
        1: "Arrow Left",
        2: "Arrow Right",
        3: "Arrow Up",
        4: "Button A",
        5: "Button B",
        6: "PASS",
        7: "PASS"
    }
    action = actions[action]
    return action