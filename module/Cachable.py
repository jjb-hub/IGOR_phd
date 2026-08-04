import os
import pickle
import shutil
from dataclasses import dataclass, field
import re
import pandas as pd



@dataclass
class Cachable:
    """Base class for caching data and managing a cache directory."""

    def __init__(self, cache_dir: str):
        """
        Initializes the Cachable class with the specified cache directory.

        Parameters:
        ----------
        cache_dir : str
            Path to the directory where cached files will be stored.
        """
        self.cache_dir = cache_dir
        self._ensure_cache_dir_exists()

    def sanitize_filename(self, identifier: str) -> str:
        illegal_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(illegal_chars, "_", identifier)
        return sanitized

    def _ensure_cache_dir_exists(self):
        """Ensures the cache directory exists; creates it if necessary."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def cache(self, identifier: str, to_cache):
        """
        Caches an object (e.g., DataFrame) as a pickle file.

        Parameters:
        ----------
        identifier : str
            Unique identifier for the cached object.
        to_cache : any
            The object to be cached.
        """
        identifier = self.sanitize_filename(identifier)
        if not identifier.endswith('.pkl'):
            identifier += '.pkl'
        cache_path = os.path.join(self.cache_dir, identifier)
        with open(cache_path, "wb") as file:
            pickle.dump(to_cache, file)
        print(f"CREATED {cache_path} CACHE")

    def getCache(self, identifier: str):
        """
        Retrieves a cached object from a pickle file.

        Parameters:
        ----------
        identifier : str
            Unique identifier for the cached object.

        Returns:
        -------
        The cached object.
        """
        cache_path = os.path.join(self.cache_dir, f"{identifier}.pkl")
        with open(cache_path, "rb") as file:
            return pickle.load(file)

    def isCached(self, identifier: str) -> bool:
        """
        Checks if a cached object exists.

        Parameters:
        ----------
        identifier : str
            Unique identifier for the cached object.

        Returns:
        -------
        bool
            True if the object is cached, False otherwise.
        """
        return os.path.isfile(os.path.join(self.cache_dir, f"{identifier}.pkl"))

    def resetCache(self):
        """
        Clears the cache directory.
        """
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)
        print("CACHE CLEARED")
        
    def _checkFileSystem(self, dir_name: str) -> str:
        """Check if the specified directory exists; if not, create it."""
        path = os.path.join(self.location, dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def save_excel(self, filename, df: pd.DataFrame):
        """
        Saves a DataFrame to an Excel file in the output directory.

        Parameters:
        ----------
        filename : str
            Base filename (with or without .xlsx).
        df : pd.DataFrame
            The DataFrame to save.
        """
        filename = self.sanitize_filename(filename)
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)
        df.to_excel(filepath, index=False, engine='openpyxl')
        print(f'CREATED {filepath} EXCEL CACHE')