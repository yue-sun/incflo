#include <incflo.H>
#include <cryo_grid.H>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

using namespace amrex;

#ifdef INCFLO_SIM_CRYO

void incflo::cryo_read_samples ()
{
    ParmParse pp("incflo");
    std::string placement;
    pp.query("cryo_sample_place", placement);
    if (placement.empty()) { return; }

    // Directory of the placement, to resolve relative .vox paths.
    std::string dir;
    {
        auto slash = placement.find_last_of('/');
        if (slash != std::string::npos) { dir = placement.substr(0, slash + 1); }
    }

    std::ifstream mfs(placement);
    if (!mfs) { amrex::Abort("cryo_read_samples: cannot open placement " + placement); }

    std::string line;
    int next_label = 1;
    while (std::getline(mfs, line))
    {
        // strip comments and skip blank lines
        auto hash = line.find('#');
        if (hash != std::string::npos) { line = line.substr(0, hash); }
        std::istringstream ls(line);
        std::string shape_file;
        if (!(ls >> shape_file)) { continue; } // blank

        Real pos_x, pos_z, angle_deg, scale;
        int flip;
        if (!(ls >> pos_x >> pos_z >> angle_deg >> flip >> scale)) {
            amrex::Abort("cryo_read_samples: malformed placement row (need: "
                         "shape_file pos_x pos_z angle_deg flip scale): " + line);
        }

        // Find or load the shape, cached by resolved path.
        std::string vox_path = dir + shape_file;
        int shape_idx = -1;
        for (int s = 0; s < static_cast<int>(m_cryo_shapes.size()); ++s) {
            if (m_cryo_shapes[s].name == vox_path) { shape_idx = s; break; }
        }
        if (shape_idx < 0) {
            VoxelShape vs;
            vs.name = vox_path;
            std::ifstream vfs(vox_path);
            if (!vfs) { amrex::Abort("cryo_read_samples: cannot open vox " + vox_path); }
            std::string vline;
            bool have_size = false, have_dims = false;
            while (std::getline(vfs, vline)) {
                auto vh = vline.find('#');
                if (vh != std::string::npos) { vline = vline.substr(0, vh); }
                std::istringstream vs_ss(vline);
                std::string tok;
                if (!(vs_ss >> tok)) { continue; }
                if (tok == "voxel_size_mm") {
                    vs_ss >> vs.voxel_size_mm; have_size = true;
                } else if (tok == "dims") {
                    vs_ss >> vs.nx >> vs.ny >> vs.nz; have_dims = true;
                    vs.occ.assign(static_cast<long>(vs.nx) * vs.ny * vs.nz, char(0));
                } else {
                    // tok is the first integer (i); j, k follow
                    if (!have_dims) { amrex::Abort("cryo_read_samples: 'dims' must precede voxels in " + vox_path); }
                    int i = std::stoi(tok), j, k;
                    if (!(vs_ss >> j >> k)) { amrex::Abort("cryo_read_samples: bad voxel line in " + vox_path); }
                    if (i >= 0 && j >= 0 && k >= 0 && i < vs.nx && j < vs.ny && k < vs.nz) {
                        vs.occ[i + vs.nx * (j + vs.ny * k)] = char(1);
                    }
                }
            }
            if (!have_size || !have_dims) { amrex::Abort("cryo_read_samples: " + vox_path + " missing voxel_size_mm or dims"); }
            shape_idx = static_cast<int>(m_cryo_shapes.size());
            m_cryo_shapes.push_back(std::move(vs));
        }

        CryoCell c;
        c.shape_idx = shape_idx;
        c.cx = pos_x; c.cz = pos_z;
        c.angle = angle_deg * Real(M_PI / 180.0);
        c.flip = (flip != 0);
        c.scale = (scale > Real(0.0)) ? scale : Real(1.0);
        c.label = next_label++;
        m_cryo_cells.push_back(c);
    }

    m_cryo_n_samples = static_cast<int>(m_cryo_cells.size());
    amrex::Print() << "cryo_read_samples: " << m_cryo_n_samples << " samples, "
                   << m_cryo_shapes.size() << " distinct shapes from " << placement << "\n";

    cryo_upload_samples();
}

// Flatten the host records into the POD form the per-cell stamper reads, and
// mirror them to device memory. Done once: the placement is fixed for the run.
// The occupancy masks of all shapes are concatenated into one buffer so a shape
// is just (dims, offset) -- no per-shape pointer to chase or upload.
void incflo::cryo_upload_samples ()
{
    int const nshapes = static_cast<int>(m_cryo_shapes.size());

    Vector<cryo_stamp::VoxelView> shapes_h(nshapes);
    Long occ_total = 0;
    for (int sidx = 0; sidx < nshapes; ++sidx)
    {
        VoxelShape const& vs = m_cryo_shapes[sidx];
        cryo_stamp::VoxelView& v = shapes_h[sidx];
        v.nx = vs.nx; v.ny = vs.ny; v.nz = vs.nz;
        v.voxel_size_mm = vs.voxel_size_mm;
        v.occ_offset = occ_total;
        occ_total += static_cast<Long>(vs.occ.size());
    }

    Vector<char> occ_h(occ_total);
    for (int sidx = 0; sidx < nshapes; ++sidx)
    {
        VoxelShape const& vs = m_cryo_shapes[sidx];
        std::copy(vs.occ.begin(), vs.occ.end(), occ_h.begin() + shapes_h[sidx].occ_offset);
    }

    Vector<cryo_stamp::CellView> cells_h(m_cryo_n_samples);
    for (int n = 0; n < m_cryo_n_samples; ++n)
    {
        CryoCell const& c = m_cryo_cells[n];
        cryo_stamp::CellView& cv = cells_h[n];
        cv.shape_idx = c.shape_idx;
        cv.cx = c.cx;
        cv.cz = c.cz;
        // The stamp un-rotates by -angle; cache that rotation here rather than
        // recomputing cos/sin for every cell of every sample.
        cv.cos_neg_angle = std::cos(-c.angle);
        cv.sin_neg_angle = std::sin(-c.angle);
        cv.flip = c.flip ? 1 : 0;
        cv.scale = c.scale;
        cv.label = c.label;
    }

    auto upload = [] (auto const& host, auto& device) {
        device.resize(host.size());
        if (!host.empty()) {
            Gpu::copyAsync(Gpu::hostToDevice, host.begin(), host.end(), device.begin());
        }
    };
    upload(shapes_h, m_cryo_shapes_d);
    upload(occ_h,    m_cryo_occ_d);
    upload(cells_h,  m_cryo_cells_d);
    Gpu::streamSynchronize();
}

cryo_stamp::SampleData incflo::cryo_sample_data (int geometry, Real plunge_disp) const
{
    cryo_stamp::SampleData sd;
    if (m_cryo_n_samples <= 0) { return sd; }   // host_valid stays false

    Real face_y, radius, zoff;
    if (!cryo_sample_grid(geometry, plunge_disp, face_y, radius, zoff)) { return sd; }

    sd.cells  = m_cryo_cells_d.dataPtr();
    sd.shapes = m_cryo_shapes_d.dataPtr();
    sd.occ    = m_cryo_occ_d.dataPtr();
    sd.n_cells = m_cryo_n_samples;
    sd.host_valid = true;
    sd.face_y = face_y;
    sd.radius = radius;
    sd.zoff   = zoff;
    return sd;
}

bool incflo::cryo_sample_grid (int geometry, Real plunge_disp,
                               Real& face_y, Real& radius, Real& zoff) const
{
    // Discrete samples sit on the +y face of any disk-family host (gold EM
    // grid / sapphire / diamond). The geometry table is the single source of
    // truth for radius and face height; see cryo_grid.H.
    cryo_grid::GridGeom grid;
    if (!cryo_grid::read_grid_geom(geometry, grid)) { return false; }

    Real const init_z = (m_cryo_disk_init_z >= Real(0.0)) ? m_cryo_disk_init_z : grid.radius;
    face_y = grid.half_thick;
    radius = grid.radius;
    zoff   = init_z + plunge_disp;
    return true;
}

#endif
