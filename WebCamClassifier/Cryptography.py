import os
import torch
from cryptography.fernet import Fernet


def load_key():
    if not os.path.isfile("secret.key"):
        generateKey()
    return open("secret.key", "rb").read()

def remove_key():
    if os.path.isfile("secret.key"):
        os.remove("secret.key")

def generateKey():
    # Generate a key (do this once and store securely)
    key = Fernet.generate_key()
    
    # Save the key to a file (keep this file private)
    with open("secret.key", "wb") as key_file:
        key_file.write(key)


def encrypt_embedding(tensor, filename):
    # Serialize the tensor to bytes
    torch.save(tensor, "temp.pt")
    with open("temp.pt", "rb") as f:
        data = f.read()
    
    # Encrypt
    fernet = Fernet(load_key())
    encrypted = fernet.encrypt(data)
    
    # Save encrypted data
    with open(filename, "wb") as f:
        f.write(encrypted)
    
    os.remove("temp.pt")  # Clean up temporary file


def decrypt_embedding(filename):
    fernet = Fernet(load_key())
    
    with open(filename, "rb") as f:
        encrypted_data = f.read()
    
    decrypted_data = fernet.decrypt(encrypted_data)
    
    # Save temporarily to load with torch
    with open("temp_decrypted.pt", "wb") as f:
        f.write(decrypted_data)
    
    embedding = torch.load("temp_decrypted.pt")
    os.remove("temp_decrypted.pt")
    return embedding
