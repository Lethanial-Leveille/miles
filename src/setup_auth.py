import os
import secrets
from pathlib import Path
from getpass import getpass
from passlib.context import CryptContext

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def main():
    print("M.I.L.E.S. v0.7 — one-time auth setup")
    print(f"Writing to: {ENV_PATH}\n")

    if ENV_PATH.exists():
        answer = input(".env already exists. Overwrite? (yes/no): ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return

    password = getpass("Set your login password: ")
    confirm  = getpass("Confirm password: ")

    if password != confirm:
        print("Passwords do not match. Aborted.")
        return

    password_hash = pwd_context.hash(password)
    jwt_secret    = secrets.token_hex(32)   # 256 bits of randomness

    ENV_PATH.write_text(
        f"MILES_PASSWORD_HASH={password_hash}\n"
        f"MILES_JWT_SECRET={jwt_secret}\n"
    )

    print("\nDone. .env written. Do not commit this file.")

if __name__ == "__main__":
    main()
