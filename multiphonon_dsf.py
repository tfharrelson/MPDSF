from src.compute_Sqw import compute_Sqw
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--poscar", help='VASP poscar file of unit cell.')
    parser.add_argument()