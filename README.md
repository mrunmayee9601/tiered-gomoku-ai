# Tiered Lightweight Submission Gomoku AI

## 📌 Overview
This project implements a **specialized AI agent for the 15×15 Gomoku game**, designed to outperform the standard Minimax baseline in both **win rate** and **runtime efficiency**.  
The AI, called **Tiered Lightweight Submission**, leverages a **hybrid tree search with tiered selective deepening** to intelligently focus on the most promising moves while respecting strict runtime constraints.  

⭐ **Academic Project**: This was a **group project** completed as part of the **Intro to AI** course, where we collaborated to design, implement, and test the AI.  

## 🛠 Tech Stack
* **Language**: Python  
* **Libraries**: NumPy (for efficient board state representation and computation)  
* **Framework/Structure**: Custom-built AI in the `Submission` class adhering to the Gomoku API

## 🚀 Methodology
The AI evaluates board positions using a heuristic that rewards potential winning patterns for itself (open-3, open-4) and penalizes patterns that help the opponent, guiding tactical decision-making efficiently.
