class GestureInterpreter:
    @staticmethod
    def is_pinch(hand_tracker, hand, img=None):
        """
        Detects if thumb and index finger are pinching.
        """
        # Tip of thumb is 4, tip of index is 8
        length, info, img = hand_tracker.get_distance(4, 8, hand, img)
        # We can use a threshold from config or a fixed one for now
        return length < 40, length, info, img

    @staticmethod
    def get_active_gesture(fingers):
        """
        Interprets the list of fingers up into a logical gesture state.
        0: Thumb, 1: Index, 2: Middle, 3: Ring, 4: Pinky
        """
        if fingers == [0, 1, 0, 0, 0]:
            return "DRAW"
        elif fingers == [0, 1, 1, 0, 0]:
            return "SELECT"
        elif fingers == [1, 1, 1, 1, 1]:
            return "ERASE_ALL"
        elif fingers == [1, 1, 0, 0, 0]:
            return "ZOOM" # Thumb and Index up
        return "IDLE"
