import importlib

import tdr

if __name__ == "__main__":
    while True:
        print("\nWhat do you want to do?")
        print("  [a] All")
        for key, (desc, _) in tdr.TASKS.items():
            print(f"  [{key}] {desc}")
        print("  [q] Quit")

        com = input("Choose: ").strip().lower()
        if com == "q":
            break
        if (com not in tdr.TASKS) and (com != "a"):
            print("Unknown choice.")
            continue

        importlib.reload(tdr)

        measurements = tdr.get_measurements()
        try:
            selected = tdr.choose_measurements(measurements)
            if com == "a":
                for key, (desc, func) in tdr.TASKS.items():
                    print(f"\n===== [{key}] {desc} =====")
                    try:
                        func(selected)
                    except Exception as e:
                        print(f"  [{key}] failed: {e}")
            else:
                tdr.TASKS[com][1](selected)
        except Exception as e:
            print("Task failed:", e)
