#ifndef CRYO_SAMPLE_
#define CRYO_SAMPLE_

// Included from cryo.H (inside the incflo class body) — declare ONLY nested
// types + members here, no #include directives. Discrete biological samples
// voxelized offline (Fiji STL -> .vox) and stamped on a disk host.

struct VoxelShape
{
    std::string name;                       // source .vox path (cache key)
    int nx = 0, ny = 0, nz = 0;
    amrex::Real voxel_size_mm = amrex::Real(0.0);
    amrex::Vector<char> occ;                // size nx*ny*nz, idx = i + nx*(j + ny*k)
    AMREX_FORCE_INLINE bool occupied (int i, int j, int k) const {
        if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return false;
        return occ[i + nx*(j + ny*k)] != 0;
    }
};

struct CryoCell
{
    int shape_idx = 0;                      // index into m_cryo_shapes
    amrex::Real cx = amrex::Real(0.0);      // disk-plane center offset x [mm]
    amrex::Real cz = amrex::Real(0.0);      // disk-plane center offset z [mm]
    amrex::Real angle = amrex::Real(0.0);   // in-plane rotation about +y [rad]
    bool flip = false;                      // mirror local x before rotation
    amrex::Real scale = amrex::Real(1.0);   // multiplies voxel_size_mm
    int label = 1;                          // cell_type (1..n)
};

int m_cryo_n_samples = 0;
amrex::Vector<VoxelShape> m_cryo_shapes;
amrex::Vector<CryoCell>   m_cryo_cells;

void cryo_read_samples ();
bool cryo_sample_grid (int geometry, amrex::Real plunge_disp,
                       amrex::Real& face_y, amrex::Real& radius,
                       amrex::Real& zoff) const;
void cryo_apply_samples (amrex::Real x, amrex::Real y, amrex::Real z,
                         amrex::Real& velx, amrex::Real& vely, amrex::Real& velz,
                         int& cell_type_ijk, amrex::Real velz_plunge,
                         amrex::Real face_y, amrex::Real radius,
                         amrex::Real zoff) const;

#endif
