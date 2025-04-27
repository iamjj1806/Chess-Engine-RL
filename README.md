# Chess AI with Reinforcement Learning

A chess application that implements a graphical interface for playing chess against an AI trained using Monte Carlo Tree Search (MCTS) and neural networks, or against the Stockfish chess engine.

## Features

- Play chess against an AI trained with MCTS and neural networks
- Play against Stockfish chess engine
- Watch AI vs Stockfish matches
- Train the AI model through self-play
- Interactive GUI with move highlighting
- Support for playing as either White or Black
- GTK-based graphical interface with SVG board rendering

## Requirements

### Python Dependencies
- Python 3.8+
- TensorFlow 2.x
- python-chess
- tkinter
- Pillow
- cairosvg
- PyGObject (GTK bindings)

### System Dependencies
- GTK 3.0+
- Cairo graphics library
- Stockfish chess engine

### Installing System Dependencies

On Windows:
```bash
# Using MSYS2
pacman -S mingw-w64-x86_64-gtk3
pacman -S mingw-w64-x86_64-cairo
```

On Ubuntu/Debian:
```bash
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0
sudo apt-get install libcairo2-dev
```

On macOS:
```bash
brew install gtk+3
brew install cairo
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ranveer1311/chess
cd chess-ai
```

2. Install required Python packages:
```bash
pip install tensorflow python-chess pillow cairosvg PyGObject
```

3. Download and install Stockfish:
   - Download Stockfish from: https://stockfishchess.org/download/
   - Extract the executable to a known location
   - Update the Stockfish path in `gui.py`

4. Create required directories:
```bash
mkdir models
mkdir memory
```

## Project Structure

- `gui.py`: GTK-based graphical interface implementation
- `mcts.py`: Monte Carlo Tree Search implementation
- `model.py`: Neural network model definition
- `training.py`: Training pipeline and self-play generation
- `environment.py`: Chess environment wrapper with extensive position evaluation
- `config.py`: Configuration settings
- `main.py`: Application entry point

## Usage

1. Start the application:
```bash
python main.py
```

2. Available options:
   - "New Game": Start a new game
   - "Train Model": Begin training the AI through self-play
   - "Play as White/Black": Start a game against the AI
   - "AI vs AI": Watch the AI play against Stockfish
   - "Play vs Stockfish": Play against Stockfish engine

## Training the AI

1. Click "Train Model" to start the training process
2. The AI will play games against itself and learn from the results
3. Training data is saved in the `memory` directory
4. Trained models are saved in the `models` directory

## Configuration

Edit `config.py` to modify:
- Neural network architecture
- Training parameters
- MCTS simulation count
- Self-play game count
- GTK interface settings
- Other hyperparameters

## Environment Features

The chess environment (`environment.py`) includes:
- Sophisticated position evaluation
- Piece-square tables for positional understanding
- Pawn structure analysis
- King safety evaluation
- Endgame recognition
- Material counting
- Mobility assessment

## Troubleshooting

### GTK Issues
- On Windows: Ensure MSYS2 is properly installed and PATH is set
- On Linux: Check if GTK development libraries are installed
- On macOS: Make sure XQuartz is installed for X11 support

### Common Problems
1. GTK Import Error:
```bash
pip install --upgrade PyGObject
```

2. Cairo SVG Rendering:
```bash
pip install --upgrade cairosvg
```

## License

[Your chosen license]

## Acknowledgments

- Stockfish chess engine
- python-chess library
- GTK Project
- Cairo graphics library
- AlphaZero paper and implementation references
