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

// Device mirrors of the two records above, in the flat POD form the per-cell
// stamper (cryo_stamp::apply_samples) consumes: one CellView per placement, one
// VoxelView per distinct shape, and a single buffer holding every shape's
// occupancy mask back to back. Built once by cryo_read_samples.
amrex::Gpu::DeviceVector<cryo_stamp::CellView>  m_cryo_cells_d;
amrex::Gpu::DeviceVector<cryo_stamp::VoxelView> m_cryo_shapes_d;
amrex::Gpu::DeviceVector<char>                  m_cryo_occ_d;

void cryo_read_samples ();
void cryo_upload_samples ();
bool cryo_sample_grid (int geometry, amrex::Real plunge_disp,
                       amrex::Real& face_y, amrex::Real& radius,
                       amrex::Real& zoff) const;
// Device-array pointers + host disk geometry, packed once per cryo_update.
// host_valid is false when this geometry hosts no samples (nothing to stamp).
cryo_stamp::SampleData cryo_sample_data (int geometry,
                                         amrex::Real plunge_disp) const;

#endif
