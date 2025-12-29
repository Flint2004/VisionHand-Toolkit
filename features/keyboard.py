import cv2

class VirtualKeyboard:
    def __init__(self):
        self.keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
                     ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
                     ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
        self.key_width = 80
        self.key_height = 80
        self.start_x = 100
        self.start_y = 100
        self.padding = 10
        self.text = ""

    def draw_keyboard(self, img, hands_info=None):
        """
        Draws the keyboard and checks for interaction.
        """
        for i, row in enumerate(self.keys):
            for j, key in enumerate(row):
                x = self.start_x + j * (self.key_width + self.padding)
                y = self.start_y + i * (self.key_height + self.padding)
                
                # Check interaction
                is_hovered = False
                if hands_info:
                    for hand in hands_info:
                        hx, hy = hand['lmList'][8][1], hand['lmList'][8][2]
                        if x < hx < x + self.key_width and y < hy < y + self.key_height:
                            is_hovered = True
                            # Optional: Check for "click" (e.g. pinch or hover duration)

                color = (0, 255, 0) if is_hovered else (255, 255, 255)
                cv2.rectangle(img, (x, y), (x + self.key_width, y + self.key_height), color, cv2.FILLED)
                cv2.putText(img, key, (x + 25, y + 55), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

        return img
