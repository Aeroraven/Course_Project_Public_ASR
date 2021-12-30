# Work 2 For Acoustic Signal Recognition


from hmm_main import AssignmentEntry

if __name__ == "__main__":
    AssignmentEntry.test_entry(
        parallel_type="mp", # (mp:Multiprocessing | mt:Multithreading | st:Single Thread)
        data_source="matlab", # (self: MFCC from Assignment 1| lib: Python MFCC library | matlab: Load MFCC files)
        use_iteration=20,
        ckpt_path="./checkpoints/"
    )