#include <incflo.H>
#include <cryo_grid.H>

#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

using namespace amrex;

#ifdef INCFLO_SIM_CRYO

// Read the height map of a measured grid (.hmap, written by
// data/grids/make_grid_hmap.py) and upload it to the device once: the grid is
// rigid, so the map never changes during the run. Only the from_map rows of
// the cryo_grid table use it; for every other geometry this is a no-op.
//
// File layout: ASCII "key value" header lines ending with "end_header", then
// two little-endian int16 planes of nz rows x nx values (x fastest), y_bottom
// then y_top, in units of height_unit_mm.
void incflo::cryo_read_grid ()
{
    ParmParse pp("incflo");
    pp.query("cryo_grid_file", m_cryo_grid_file);

    cryo_grid::GridGeom row;
    bool const from_map = cryo_grid::read_grid_geom(m_cryo_geometry, row) && row.from_map;
    if (!from_map) {
        if (!m_cryo_grid_file.empty()) {
            amrex::Abort("cryo_read_grid: incflo.cryo_grid_file is set, but cryo_geometry "
                         + std::to_string(m_cryo_geometry) + " does not read a grid map");
        }
        return;
    }
    std::string const hint = " (build it with install.sh, or: uv run python "
                             "data/grids/make_grid_hmap.py <grid>.stl)";
    if (m_cryo_grid_file.empty()) {
        amrex::Abort("cryo_read_grid: cryo_geometry " + std::to_string(m_cryo_geometry)
                     + " needs incflo.cryo_grid_file, a .hmap height map" + hint);
    }

    std::ifstream f(m_cryo_grid_file, std::ios::binary);
    if (!f) { amrex::Abort("cryo_read_grid: cannot open " + m_cryo_grid_file + hint); }

    std::map<std::string, std::string> hdr;
    std::string line;
    bool ended = false;
    while (std::getline(f, line)) {
        if (line == "end_header") { ended = true; break; }
        if (line.empty() || line[0] == '#') { continue; }
        std::istringstream ls(line);
        std::string key, val;
        ls >> key;
        std::getline(ls >> std::ws, val);
        hdr[key] = val;
    }
    if (!ended) { amrex::Abort("cryo_read_grid: no end_header in " + m_cryo_grid_file); }
    auto get = [&] (std::string const& key) -> std::string const& {
        auto it = hdr.find(key);
        if (it == hdr.end()) { amrex::Abort("cryo_read_grid: " + m_cryo_grid_file + " has no '" + key + "'"); }
        return it->second;
    };
    if (get("format") != "cryoflo-hmap 1") {
        amrex::Abort("cryo_read_grid: unsupported format '" + get("format") + "' in " + m_cryo_grid_file);
    }

    cryo_grid::GridMap m;
    m.nx     = std::stoi(get("nx"));
    m.nz     = std::stoi(get("nz"));
    m.x0     = std::stod(get("x0_mm"));
    m.z0     = std::stod(get("z0_mm"));
    m.pixel  = std::stod(get("pixel_mm"));
    m.unit   = std::stod(get("height_unit_mm"));
    m.empty  = static_cast<short>(std::stoi(get("empty")));
    m.radius = std::stod(get("radius_mm"));
    m.face_y = std::stod(get("face_y_mm"));
    if (m.nx <= 0 || m.nz <= 0 || m.pixel <= 0.0 || m.unit <= 0.0 || m.radius <= 0.0) {
        amrex::Abort("cryo_read_grid: bad header in " + m_cryo_grid_file);
    }

    // The planes are little-endian int16; read them straight into `short`.
    static_assert(sizeof(short) == 2, "cryo_read_grid: needs a 16-bit short");
    std::uint16_t const probe = 1;
    if (*reinterpret_cast<unsigned char const*>(&probe) != 1) {
        amrex::Abort("cryo_read_grid: big-endian hosts are not supported");
    }
    Long const npix = static_cast<Long>(m.nx) * m.nz;
    Vector<short> planes(2 * npix);
    std::streamsize const nbytes = static_cast<std::streamsize>(2 * npix * sizeof(short));
    f.read(reinterpret_cast<char*>(planes.data()), nbytes);
    if (f.gcount() != nbytes) {
        amrex::Abort("cryo_read_grid: " + m_cryo_grid_file + " is truncated");
    }
    if (f.peek() != std::char_traits<char>::eof()) {
        amrex::Abort("cryo_read_grid: unexpected trailing data in " + m_cryo_grid_file);
    }

    m_cryo_grid_hmap_d.resize(planes.size());
    Gpu::copyAsync(Gpu::hostToDevice, planes.begin(), planes.end(), m_cryo_grid_hmap_d.begin());
    Gpu::streamSynchronize();
    m.y_bottom = m_cryo_grid_hmap_d.dataPtr();
    m.y_top    = m.y_bottom + npix;
    m_cryo_grid_map = m;

    amrex::Print() << "cryo_read_grid: " << m_cryo_grid_file << ": " << m.nx << " x " << m.nz
                   << " pixels of " << m.pixel * 1.0e3 << " um, radius " << m.radius
                   << " mm, top face y = " << m.face_y * 1.0e3 << " um ("
                   << get("source") << ")\n";
}

bool incflo::cryo_grid_geom (int geometry, cryo_grid::GridGeom& grid) const
{
    if (!cryo_grid::read_grid_geom(geometry, grid)) { return false; }
    if (grid.from_map) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_cryo_grid_map.y_bottom != nullptr,
                                         "cryo_grid_geom: grid map not loaded (cryo_read_grid)");
        grid.map    = m_cryo_grid_map;
        grid.radius = m_cryo_grid_map.radius;
        grid.face_y = m_cryo_grid_map.face_y;
    }
    return true;
}

#endif
