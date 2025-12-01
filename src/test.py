from desmume.emulator import DeSmuME
import time

def main():
    emu = DeSmuME()
    emu.open("ROM/mariokart.nds")
     
    window = emu.create_sdl_window()
    while not window.has_quit():
        window.process_input()
        time.sleep(0.01)
        emu.cycle()
        window.draw()

if __name__ == "__main__":
    main()
