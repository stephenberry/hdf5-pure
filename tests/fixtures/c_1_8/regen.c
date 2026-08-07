/* Regenerate the HDF5 1.8-era read fixtures. See NOTICE.md for why these are
 * produced by an old library rather than by the `hdf5-metno` dev-dependency.
 *
 * Built and run by `regen.sh`, which points it at the HDF5 1.8.23 install that
 * `scripts/check-hdf5-18.sh` leaves under `tmp/hdf5-18-check/install`.
 */
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                              \
    do {                                                                      \
        if ((x) < 0) {                                                        \
            fprintf(stderr, "failed at %s:%d: %s\n", __FILE__, __LINE__, #x); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

/* A scalar ASCII-string attribute, the shape every fixture here carries so the
 * reader has to walk an object header rather than only a superblock. */
static void put_attr(hid_t obj, const char *name, const char *value) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t ty = H5Tcopy(H5T_C_S1);
    CHECK(H5Tset_size(ty, strlen(value)));
    hid_t at = H5Acreate2(obj, name, ty, sp, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(at);
    CHECK(H5Awrite(at, ty, value));
    CHECK(H5Aclose(at));
    CHECK(H5Tclose(ty));
    CHECK(H5Sclose(sp));
}

static void put_contiguous(hid_t where, const char *name) {
    hsize_t dims[1] = {4};
    double v[4] = {1.5, 2.5, 3.5, 4.5};
    hid_t sp = H5Screate_simple(1, dims, NULL);
    hid_t ds = H5Dcreate2(where, name, H5T_IEEE_F64LE, sp, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT);
    CHECK(ds);
    CHECK(H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v));
    put_attr(ds, "units", "m/s");
    CHECK(H5Dclose(ds));
    CHECK(H5Sclose(sp));
}

/* Chunked and deflated. Under a pre-1.10 superblock this is indexed by a
 * version 1 B-tree, the index this crate reads and does not write. */
static void put_chunked(hid_t where, const char *name) {
    hsize_t dims[1] = {1000}, chunk[1] = {100};
    int *v = malloc(1000 * sizeof(int));
    for (int i = 0; i < 1000; i++) v[i] = i % 97;
    hid_t sp = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    CHECK(H5Pset_chunk(dcpl, 1, chunk));
    CHECK(H5Pset_deflate(dcpl, 6));
    hid_t ds = H5Dcreate2(where, name, H5T_STD_I32LE, sp, H5P_DEFAULT, dcpl,
                          H5P_DEFAULT);
    CHECK(ds);
    CHECK(H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, v));
    CHECK(H5Dclose(ds));
    CHECK(H5Pclose(dcpl));
    CHECK(H5Sclose(sp));
    free(v);
}

static void fill(hid_t f) {
    put_contiguous(f, "values");
    put_chunked(f, "chunked");
    hid_t g = H5Gcreate2(f, "grp", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(g);
    put_attr(g, "tag", "group");
    put_contiguous(g, "inner");
    CHECK(H5Gclose(g));
    put_attr(f, "root_attr", "r");
}

int main(void) {
    /* A version 1 superblock. The C library writes version 0 by default and
     * bumps to 1 exactly when a B-tree K value is non-default — those K values
     * are the fields the version 1 layout adds, and reading them from the wrong
     * offsets is the defect this fixture exists to catch. */
    {
        hid_t fcpl = H5Pcreate(H5P_FILE_CREATE);
        CHECK(H5Pset_istore_k(fcpl, 64)); /* default 32 */
        CHECK(H5Pset_sym_k(fcpl, 8, 16)); /* defaults 16 (internal), 4 (leaf) */
        hid_t f = H5Fcreate("v1_superblock.h5", H5F_ACC_TRUNC, fcpl, H5P_DEFAULT);
        CHECK(f);
        fill(f);
        CHECK(H5Fclose(f));
        CHECK(H5Pclose(fcpl));
    }

    /* A version 2 superblock: 1.8's newest format, and the one this crate now
     * writes by default for `.mat` files. */
    {
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        CHECK(H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST));
        hid_t f = H5Fcreate("v2_superblock.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        CHECK(f);
        fill(f);
        CHECK(H5Fclose(f));
        CHECK(H5Pclose(fapl));
    }

    printf("wrote v1_superblock.h5 and v2_superblock.h5\n");
    return 0;
}
