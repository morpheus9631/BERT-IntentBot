import os

class PathUtil():

    _RootPath = None

    # -------------------------------------------------------------------------

    @classmethod
    def get_root_path(cls) -> str:
        """回傳專案的根路徑"""
        if cls._RootPath is None:
            curr_path     = os.path.dirname(os.path.abspath(__file__))
            root_path_str = os.path.join(curr_path, '../../../../')
            root_path     = os.path.abspath(root_path_str)
            cls._RootPath = root_path.lower()

        return cls._RootPath

    # -------------------------------------------------------------------------
    
    @staticmethod
    def ensure_path_exists(target_path):
        """當路徑不存在時會建立新的，以確保路徑存在"""
        try:
            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)
                print(f"'{target_path}' path not exists, already created.")
            return True
        except Exception as ex: 
            print(f"Failed to ensure path exists: {ex}")
            return False
        
    # -------------------------------------------------------------------------
    
    @staticmethod
    def get_parent_path(file_name: str) -> str:
        current_abspath = os.path.abspath(file_name)
        parent_path = os.path.dirname(current_abspath)

        return parent_path

    # -------------------------------------------------------------------------
        