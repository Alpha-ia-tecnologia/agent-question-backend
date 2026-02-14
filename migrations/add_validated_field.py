"""
Migration: Add validated field to questions table
Run this SQL against your database to add the validated column.
"""

# SQL Migration
SQL_UPGRADE = """
ALTER TABLE questions 
ADD COLUMN IF NOT EXISTS validated BOOLEAN DEFAULT FALSE;
"""

SQL_DOWNGRADE = """
ALTER TABLE questions 
DROP COLUMN IF EXISTS validated;
"""

if __name__ == "__main__":
    print("=== Migration: Add validated field to questions ===")
    print()
    print("Run this SQL command on your PostgreSQL database:")
    print()
    print(SQL_UPGRADE)
    print()
    print("To rollback, run:")
    print()
    print(SQL_DOWNGRADE)
