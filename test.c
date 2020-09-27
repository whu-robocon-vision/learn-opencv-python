#include <stdio.h>

int main(int argc, char const *argv[])
{
    int h = 10;

    for (int w = 1; w < 1000; w++) {
        if (w / h > 0.5
        && (h / w) > 0.5
        )
        {
            printf("hahhaha %d %d", w, h);
            return 0;
        }
    }
    printf("xiixixi");
    return 0;
}
