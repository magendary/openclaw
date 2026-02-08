Avoid a file that just re-exports from another file. Instead, import directly from the original file. This helps to reduce unnecessary indirection and makes it easier to understand where the code is coming from.

Avoid redundancies and inconsistencies. If a function already exists in one file, do not create a new function that does the same thing in another file. Instead, import the existing function and use it. This helps to keep the codebase clean and maintainable.
