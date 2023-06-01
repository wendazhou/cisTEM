#pragma once

#include <utility>

/** Utility wrapper class for compatibility when accesing the atom positions
 * 
 */
class AtomPosArrayWrapper {
  private:
    struct AtomPosReference {
        float& x;
        float& y;
        float& z;
        float& w;
    };

    struct AtomPosConstReference {
        const float& x;
        const float& y;
        const float& z;
        const float& w;
    };

    float* positions_;

  public:
    AtomPosArrayWrapper(float* positions) : positions_(positions) {}

    AtomPosReference operator[](int index) {
        return AtomPosReference{positions_[index * 4 + 0], positions_[index * 4 + 1], positions_[index * 4 + 2], positions_[index * 4 + 3]};
    }

    AtomPosConstReference operator[](int index) const {
        return AtomPosConstReference{positions_[index * 4 + 0], positions_[index * 4 + 1], positions_[index * 4 + 2], positions_[index * 4 + 3]};
    }
};

class Water {
  private:
    //! Array containing the water positions
    float* water_positions_ = nullptr;
    //! Array containing the transformed water positions, sorted by z
    float* transformed_water_positions_ = nullptr;

  public:
    // Constructors
    Water(bool do_carbon = false);
    Water(const PDB* current_specimen, int wanted_size_neighborhood, float wanted_pixel_size, float wanted_dose_per_frame, RotationMatrix max_rotation, float in_plane_rotation, int* padX, int* padY, int nThreads, bool pad_based_on_rotation, bool do_carbon = false);
    ~Water( );

    // data

    long                number_of_waters = 0;
    int                 size_neighborhood;
    float               pixel_size;
    float               dose_per_frame;
    int                 records_per_line;
    int                 number_of_time_steps;
    bool                simulate_phase_plate;
    bool                keep_time_steps;
    double              center_of_mass[3];
    float               atomic_volume;
    float               vol_angX, vol_angY, vol_angZ;
    int                 vol_nX, vol_nY, vol_nZ;
    float               vol_oX, vol_oY, vol_oZ;
    float               offset_z;
    float               min_z;
    float               max_z;
    AtomPosArrayWrapper water_coords;
    bool                is_allocated_water_coords = false;
    int                 nThreads;

    //	void set_initial_trajectories(PDB *pdb_ensemble);

    void Init(const PDB* current_specimen, int wanted_size_neighborhood, float wanted_pixel_size, float wanted_dose_per_frame, RotationMatrix max_rotation, float in_plane_rotation, int* padX, int* padY, int nThreads, bool pad_based_on_rotation);
    void SeedWaters3d( );
    void ShakeWaters3d(int number_of_threads);
    void ReturnPadding(RotationMatrix max_rotation, float in_plane_rotation, int current_thickness, int current_nX, int current_nY, int* padX, int* padY, int* padZ);

    // Fills the array of transformed positions according to the given rotation.
    void ComputeTransformedPositions(RotationMatrix rotation);
    // Sorts the transformed position by z index
    void SortTransformedPositions();
    // Obtain a pointer to the transformed positions
    // This array represents the transformed position of each water molecule, where there are number_of_waters such
    // Each water molecule is represented by 4 floats, x, y, z, and w.
    float const* GetTransformedPositions() const { return transformed_water_positions_; }
    // Assuming that the transformed positions have been sorted along the z direction,
    // Compute the range of indices that correspond to the slab defined by the given z coordinates.
    // The indices are returned as a half-open interval [start_idx, end_idx)
    std::pair<std::size_t, std::size_t> GetSlabRange(float slab_z_start, float slab_z_end) const;

    inline float Return_x_Coordinate(long current_atom) { return water_coords[current_atom].x; }

    inline void ReturnCenteredCoordinates(long current_atom, float& dx, float& dy, float& dz) {

        // Find the water's coordinates in the rotated slab
        // Shift those coordinates to a centered origin
        dx = water_coords[current_atom].x - vol_oX;
        dy = water_coords[current_atom].y - vol_oY;
        dz = water_coords[current_atom].z - vol_oZ;
    }
};
