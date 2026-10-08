import importlib

import tdr

if __name__ == "__main__":
    while True:
        print("\nWhat do you want to do?")
        for key, (desc, _) in tdr.TASKS.items():
            print(f"  [{key}] {desc}")
        print("  [q] Quit")

        com = input("Choose: ").strip().lower()
        if com == "q":
            break
        if com not in tdr.TASKS:
            print("Unknown choice.")
            continue

        importlib.reload(tdr)

        measurements = tdr.get_measurements()
        try:
            tdr.TASKS[com][1](tdr.choose_measurements(measurements))
        except Exception as e:
            print("Task failed:", e)
