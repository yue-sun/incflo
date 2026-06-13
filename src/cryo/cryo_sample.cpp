#include <incflo.H>

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
}

bool incflo::cryo_sample_grid (int geometry, Real plunge_disp,
                               Real& face_y, Real& radius, Real& zoff) const
{
    Real R, w;
    if (geometry == 4) { R = Real(1.5); w = Real(0.16 / 2); }       // sapphire disk
    else if (geometry == 5) { R = Real(1.5); w = Real(0.1 / 2); }   // diamond disk
    else { return false; }                                          // (gold grid: future)

    Real const init_z = (m_cryo_disk_init_z >= Real(0.0)) ? m_cryo_disk_init_z : R;
    face_y = w;
    radius = R;
    zoff   = init_z + plunge_disp;
    return true;
}

void incflo::cryo_apply_samples (Real x, Real y, Real z,
                                 Real& velx, Real& vely, Real& velz,
                                 int& cell_type_ijk, Real velz_plunge,
                                 Real face_y, Real radius, Real zoff) const
{
    // Off the disk footprint: nothing to host.
    Real const rz = z - zoff;
    if (x * x + rz * rz > radius * radius) { return; }
    Real const ly = y - face_y;
    if (ly < Real(0.0)) { return; }

    for (int n = 0; n < m_cryo_n_samples; ++n)
    {
        CryoCell const& c = m_cryo_cells[n];
        VoxelShape const& s = m_cryo_shapes[c.shape_idx];
        Real const v = c.scale * s.voxel_size_mm;
        if (ly > Real(s.ny) * v) { continue; }      // above this cell's height

        // local in-plane coords relative to the cell center
        Real lx = x - c.cx;
        Real lz = rz - c.cz;
        // un-rotate by -angle about +y
        Real const ca = std::cos(-c.angle), sa = std::sin(-c.angle);
        Real lxr = ca * lx - sa * lz;
        Real const lzr = sa * lx + ca * lz;
        if (c.flip) { lxr = -lxr; }

        // to voxel indices (centered in i,k; j from the face)
        int const vi = static_cast<int>(std::floor(lxr / v + Real(0.5) * s.nx));
        int const vj = static_cast<int>(std::floor(ly  / v));
        int const vk = static_cast<int>(std::floor(lzr / v + Real(0.5) * s.nz));

        if (s.occupied(vi, vj, vk))
        {
            cell_type_ijk = c.label;
            velx = Real(0.0);
            vely = Real(0.0);
            velz = velz_plunge;      // rigid, co-moves with the disk
            return;                  // first match (placement order) wins
        }
    }
}

#endif
