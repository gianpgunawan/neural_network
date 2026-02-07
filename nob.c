#define NOB_IMPLEMENTATION
#include "nob.h"

void build_main(int argc, char **argv)
{
    Nob_Cmd cmd = {0};
    
#ifdef _WIN32
    const char *out = ".\\main.exe";
#else
    const char *out = "./main";
#endif // _WIN32
    Nob_File_Paths source_paths = {0};
    nob_da_append(&source_paths, "main.c");
    nob_da_append(&source_paths, "matrix.h");
    nob_da_append(&source_paths, "nn.h"); nob_da_append(&source_paths, "nn.c");
    nob_da_append(&source_paths, "arena.h");
    nob_da_append(&source_paths, "main.c");
    if (nob_needs_rebuild(out, source_paths.items, source_paths.count)) {
        nob_cmd_append(&cmd, "gcc", "-o", "main", "main.c", "-lm", "-O3");
        if (!nob_cmd_run(&cmd)) exit(1);
    }
#ifdef _WIN32
    nob_cmd_append(&cmd, ".\\main.exe");
    nob_shift(argv, argc);
    nob_cmd_append(&cmd, nob_shift(argv, argc));
    nob_cmd_append(&cmd, nob_shift(argv, argc));
#else 
    nob_cmd_append(&cmd, "./main");
#endif // _WIN32
    if (!nob_cmd_run(&cmd)) exit(1);
}

int main(int argc, char **argv)
{
    NOB_GO_REBUILD_URSELF(argc, argv);
    build_main(argc, argv);
}
