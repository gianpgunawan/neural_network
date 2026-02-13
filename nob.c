#define NOB_IMPLEMENTATION
#include "nob.h"
#include <string.h>

void build_main(int argc, char **argv)
{
    Nob_Cmd cmd = {0};
#ifdef _WIN32
    const char *BIN_DIR = ".\\bin\\";
    const char *MAIN = "main.exe";
#else
    const char *BIN_DIR = "./bin/";
    const char *MAIN = "main";
#endif // _WIN32
    
    String_Builder target_sb = {0};   
    sb_append_cstr(&target_sb, BIN_DIR);
    sb_append_cstr(&target_sb, MAIN);
    const char *TARGET = temp_sv_to_cstr(sb_to_sv(target_sb));

    nob_cmd_append(&cmd, "make", "all");
    if (!nob_cmd_run(&cmd)) exit(1);
    nob_cmd_append(&cmd, TARGET);
    nob_shift(argv, argc);
    nob_cmd_append(&cmd, nob_shift(argv, argc));
    nob_cmd_append(&cmd, nob_shift(argv, argc));
    if (!nob_cmd_run(&cmd)) exit(1);
}

bool walk(Nob_Walk_Entry entry)
{
    String_View sv_entrypath = sv_from_cstr(entry.path);
    bool skip = false;
    skip = skip || (sv_starts_with(sv_entrypath, sv_from_cstr(".\\.git")));
    skip = skip || (sv_starts_with(sv_entrypath, sv_from_cstr(".\\bin")));
    skip = skip || (sv_starts_with(sv_entrypath, sv_from_cstr(".\\tags")));
    if (skip) return true;
    nob_log(NOB_WARNING, entry.path);
    return true;
}

int experiment(int argc, char **argv)
{
    NOB_GO_REBUILD_URSELF(argc, argv);

    Nob_Walk_Dir_Opt opt = {
        .data = (char *)"hello",
        .post_order = false,
    };

    walk_dir(".\\includes", walk, &opt);
    return 0;
}

int main(int argc, char **argv)
{
    NOB_GO_REBUILD_URSELF(argc, argv);
    build_main(argc, argv);
}
