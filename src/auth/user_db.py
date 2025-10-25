"""
User database and authentication management.

Handles user accounts and analysis history storage.
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import base64


class UserDatabase:
    """Manages user accounts and analysis history."""

    def __init__(self, db_path: str = "data/users.db"):
        """Initialize database connection."""
        try:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.init_database()
        except Exception as e:
            # Fallback to temp directory if data directory not writable
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "modium_db"
            temp_dir.mkdir(exist_ok=True)
            self.db_path = temp_dir / "users.db"
            self.init_database()

    def init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Analysis history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_data TEXT,
                predicted_class TEXT,
                confidence REAL,
                risk_score REAL,
                risk_level TEXT,
                visual_features TEXT,
                recommendations TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        conn.commit()
        conn.close()

    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username: str, password: str, email: str = None) -> Tuple[bool, str]:
        """
        Create a new user account.

        Returns:
            (success, message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            password_hash = self.hash_password(password)

            cursor.execute(
                'INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)',
                (username, password_hash, email)
            )

            conn.commit()
            conn.close()
            return True, "Account created successfully!"

        except sqlite3.IntegrityError:
            return False, "Username already exists"
        except Exception as e:
            return False, f"Error creating account: {str(e)}"

    def verify_user(self, username: str, password: str) -> Tuple[bool, Optional[int]]:
        """
        Verify user credentials.

        Returns:
            (success, user_id)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        password_hash = self.hash_password(password)

        cursor.execute(
            'SELECT id FROM users WHERE username = ? AND password_hash = ?',
            (username, password_hash)
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return True, result[0]
        return False, None

    def save_analysis(
        self,
        user_id: int,
        image_base64: str,
        predicted_class: str,
        confidence: float,
        risk_score: float,
        risk_level: str,
        visual_features: Dict,
        recommendations: List[str]
    ) -> bool:
        """Save an analysis to user's history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO analyses
                (user_id, image_data, predicted_class, confidence, risk_score,
                 risk_level, visual_features, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                image_base64,
                predicted_class,
                confidence,
                risk_score,
                risk_level,
                json.dumps(visual_features),
                json.dumps(recommendations)
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"Error saving analysis: {e}")
            return False

    def get_user_analyses(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get analysis history for a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, timestamp, predicted_class, confidence, risk_score,
                   risk_level, visual_features, recommendations, image_data
            FROM analyses
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))

        results = cursor.fetchall()
        conn.close()

        analyses = []
        for row in results:
            analyses.append({
                'id': row[0],
                'timestamp': row[1],
                'predicted_class': row[2],
                'confidence': row[3],
                'risk_score': row[4],
                'risk_level': row[5],
                'visual_features': json.loads(row[6]),
                'recommendations': json.loads(row[7]),
                'image_data': row[8]
            })

        return analyses

    def get_user_stats(self, user_id: int) -> Dict:
        """Get statistics for a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total analyses
        cursor.execute('SELECT COUNT(*) FROM analyses WHERE user_id = ?', (user_id,))
        total = cursor.fetchone()[0]

        # Risk level distribution
        cursor.execute('''
            SELECT risk_level, COUNT(*)
            FROM analyses
            WHERE user_id = ?
            GROUP BY risk_level
        ''', (user_id,))
        risk_dist = dict(cursor.fetchall())

        # Average risk score
        cursor.execute('''
            SELECT AVG(risk_score)
            FROM analyses
            WHERE user_id = ?
        ''', (user_id,))
        avg_risk = cursor.fetchone()[0] or 0

        conn.close()

        return {
            'total_analyses': total,
            'risk_distribution': risk_dist,
            'average_risk_score': avg_risk
        }
