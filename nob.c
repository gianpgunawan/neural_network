#define NOB_IMPLEMENTATION
#define NOB_STRIP_PREFIX
#include "nob.h"

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

int main(int argc, char **argv)
{
    NOB_GO_REBUILD_URSELF(argc, argv);
    build_main(argc, argv);
}
