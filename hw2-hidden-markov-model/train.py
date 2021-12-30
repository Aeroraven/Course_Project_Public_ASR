# Work 2 For Acoustic Signal Recognition


from hmm_main import AssignmentEntry

if __name__ == "__main__":
    AssignmentEntry.train_entry(
        parallel_type="mt", # (mp:Multiprocessing | mt:Multithreading | st:Single Thread)
        data_source="matlab", # (self: MFCC from Assignment 1| lib: Python MFCC library | matlab: Load MFCC files)
        val_split=0.8,
        iterations=20,
        save_path="./checkpoints"
    )