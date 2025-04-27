import tkinter as tk
from config import Config
from gui import ChessGUI

def main():
    # Create necessary directories
    Config.create_directories()
    
    # Create GUI
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
