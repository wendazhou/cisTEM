#include "../../core/core_headers.h"

class
        CalculateCrossCorrelationsApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

Peak get_optimal_peak(Image* input_image, Image* template_reconstruction, Image* projection_filter, float psi_opt, float theta_opt, float phi_opt, float padding) {
    AnglesAndShifts angles;
    angles.Init(phi_opt, theta_opt, psi_opt, 0.0, 0.0);

    Image padded_projection;
    if ( padding != 1.0f )
        padded_projection.Allocate(template_reconstruction->logical_x_dimension * padding, template_reconstruction->logical_y_dimension * padding, false);

    Image current_projection;
    current_projection.Allocate(template_reconstruction->logical_x_dimension, template_reconstruction->logical_x_dimension, false);

    Image padded_reference;
    padded_reference.Allocate(input_image->logical_x_dimension, input_image->logical_y_dimension, 1);
    padded_reference.SetToConstant(0.0f);

    if ( padding != 1.0f ) {
        template_reconstruction->ExtractSlice(padded_projection, angles, 1.0f, false);
        padded_projection.SwapRealSpaceQuadrants( );
        padded_projection.BackwardFFT( );
        padded_projection.ClipInto(&current_projection);
        current_projection.ForwardFFT( );
    }
    else {
        template_reconstruction->ExtractSlice(current_projection, angles, 1.0f, false);
        current_projection.SwapRealSpaceQuadrants( );
    }

    current_projection.MultiplyPixelWise(*projection_filter);

    current_projection.BackwardFFT( );
    current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges( ));
    float variance = current_projection.ReturnSumOfSquares( ) * current_projection.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels - powf(current_projection.ReturnAverageOfRealValues( ) * current_projection.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels, 2);
    current_projection.DivideByConstant(sqrtf(variance));
    current_projection.ClipIntoLargerRealSpace2D(&padded_reference);

    padded_reference.ForwardFFT( );
    padded_reference.ZeroCentralPixel( );

#ifdef MKL
    // Use the MKL
    vmcMulByConj(padded_reference.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(input_image->complex_values), reinterpret_cast<MKL_Complex8*>(padded_reference.complex_values), reinterpret_cast<MKL_Complex8*>(padded_reference.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    for ( pixel_counter = 0; pixel_counter < padded_reference.real_memory_allocated / 2; pixel_counter++ ) {
        padded_reference.complex_values[pixel_counter] = conj(padded_reference.complex_values[pixel_counter]) * input_image.complex_values[pixel_counter];
    }
#endif
    padded_reference.BackwardFFT( );
    wxPrintf("max cc = %f", padded_reference.ReturnMaximumValue( ));
    return padded_reference.FindPeakWithIntegerCoordinates( );
}

IMPLEMENT_APP(CalculateCrossCorrelationsApp)

// override the DoInteractiveUserInput

void CalculateCrossCorrelationsApp::DoInteractiveUserInput( ) {
    wxString input_search_images;
    wxString input_reconstruction;

    wxString output_histogram_file;
    wxString correlation_avg_output_file;
    wxString correlation_std_output_file;

    float pixel_size              = 1.0f;
    float voltage_kV              = 300.0f;
    float spherical_aberration_mm = 2.7f;
    float amplitude_contrast      = 0.07f;
    float defocus1                = 10000.0f;
    float defocus2                = 10000.0f;

    float    defocus_angle;
    float    phase_shift;
    float    low_resolution_limit      = 300.0;
    float    high_resolution_limit     = 8.0;
    float    angular_step              = 5.0;
    int      best_parameters_to_keep   = 20;
    float    defocus_search_range      = 500;
    float    defocus_step              = 50;
    float    pixel_size_search_range   = 0.1f;
    float    pixel_size_step           = 0.02f;
    float    padding                   = 1.0;
    float    particle_radius_angstroms = 0.0f;
    wxString my_symmetry               = "C1";
    float    in_plane_angular_step     = 0;
    int      max_threads               = 1; // Only used for the GPU code
    float    psi_opt;
    float    theta_opt;
    float    phi_opt;

    UserInput* my_input = new UserInput("SaveCrossCorrelations", 1.00);

    input_search_images   = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "image_stack.mrc", true);
    input_reconstruction  = my_input->GetFilenameFromUser("Input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    output_histogram_file = my_input->GetFilenameFromUser("Output histogram of correlation values", "histogram of all correlation values", "histogram.txt", false);
    //correlation_avg_output_file = my_input->GetFilenameFromUser("Correlation average value", "The file for saving the average value of all correlation images", "corr_average.mrc", false);
    //correlation_std_output_file = my_input->GetFilenameFromUser("Correlation standard deviation value", "The file for saving the std value of all correlation images", "corr_variance.mrc", false);
    pixel_size              = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    voltage_kV              = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
    spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7");
    amplitude_contrast      = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
    defocus1                = my_input->GetFloatFromUser("Defocus1 (angstroms)", "Defocus1 for the input image", "10000", 0.0);
    defocus2                = my_input->GetFloatFromUser("Defocus2 (angstroms)", "Defocus2 for the input image", "10000", 0.0);
    defocus_angle           = my_input->GetFloatFromUser("Defocus Angle (degrees)", "Defocus Angle for the input image", "0.0");
    phase_shift             = my_input->GetFloatFromUser("Phase Shift (degrees)", "Additional phase shift in degrees", "0.0");
    //    low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
    high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
    angular_step          = my_input->GetFloatFromUser("Out of plane angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
    in_plane_angular_step = my_input->GetFloatFromUser("In plane angular step (0.0 = set automatically)", "Angular step size for in-plane rotations during the search", "0.0", 0.0);
    //    best_parameters_to_keep = my_input->GetIntFromUser("Number of top hits to refine", "The number of best global search orientations to refine locally", "20", 1);
    padding                   = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0, 2.0);
    particle_radius_angstroms = my_input->GetFloatFromUser("Mask radius for global search (A) (0.0 = max)", "Radius of a circular mask to be applied to the input images during global search", "0.0", 0.0);
    my_symmetry               = my_input->GetSymmetryFromUser("Template symmetry", "The symmetry of the template reconstruction", "C1");
    phi_opt                   = my_input->GetFloatFromUser("Optimal phi", "Optimal phi", "0.0", 0.0, 360.0);
    theta_opt                 = my_input->GetFloatFromUser("Optimal theta", "Optimal theta", "0.0", 0.0, 180.0);
    psi_opt                   = my_input->GetFloatFromUser("Optimal psi", "Optimal psi", "0.0", 0.0, 360.0);

    int   first_search_position = -1;
    int   last_search_position  = -1;
    float min_peak_radius       = 10.0f;

    delete my_input;

    my_current_job.ManualSetArguments("ttffffffffffiffffffftftiiffff", input_search_images.ToUTF8( ).data( ),
                                      input_reconstruction.ToUTF8( ).data( ),
                                      pixel_size,
                                      voltage_kV,
                                      spherical_aberration_mm,
                                      amplitude_contrast,
                                      defocus1,
                                      defocus2,
                                      defocus_angle,
                                      low_resolution_limit,
                                      high_resolution_limit,
                                      angular_step,
                                      best_parameters_to_keep,
                                      defocus_search_range,
                                      defocus_step,
                                      pixel_size_search_range,
                                      pixel_size_step,
                                      padding,
                                      particle_radius_angstroms,
                                      phase_shift,
                                      my_symmetry.ToUTF8( ).data( ),
                                      in_plane_angular_step,
                                      output_histogram_file.ToUTF8( ).data( ),
                                      first_search_position,
                                      last_search_position,
                                      min_peak_radius,
                                      phi_opt,
                                      theta_opt,
                                      psi_opt
                                      //correlation_avg_output_file.ToUTF8( ).data( ),
                                      //correlation_std_output_file.ToUTF8( ).data( )

    );
}

// override the do calculation method which will be what is actually run..

bool CalculateCrossCorrelationsApp::DoCalculation( ) {
    wxString input_search_images_filename  = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_reconstruction_filename = my_current_job.arguments[1].ReturnStringArgument( );
    float    pixel_size                    = my_current_job.arguments[2].ReturnFloatArgument( );
    float    voltage_kV                    = my_current_job.arguments[3].ReturnFloatArgument( );
    float    spherical_aberration_mm       = my_current_job.arguments[4].ReturnFloatArgument( );
    float    amplitude_contrast            = my_current_job.arguments[5].ReturnFloatArgument( );
    float    defocus1                      = my_current_job.arguments[6].ReturnFloatArgument( );
    float    defocus2                      = my_current_job.arguments[7].ReturnFloatArgument( );
    float    defocus_angle                 = my_current_job.arguments[8].ReturnFloatArgument( );
    float    low_resolution_limit          = my_current_job.arguments[9].ReturnFloatArgument( );
    float    high_resolution_limit_search  = my_current_job.arguments[10].ReturnFloatArgument( );
    float    angular_step                  = my_current_job.arguments[11].ReturnFloatArgument( );
    int      best_parameters_to_keep       = my_current_job.arguments[12].ReturnIntegerArgument( );
    float    defocus_search_range          = my_current_job.arguments[13].ReturnFloatArgument( );
    float    defocus_step                  = my_current_job.arguments[14].ReturnFloatArgument( );
    float    pixel_size_search_range       = my_current_job.arguments[15].ReturnFloatArgument( );
    float    pixel_size_step               = my_current_job.arguments[16].ReturnFloatArgument( );
    float    padding                       = my_current_job.arguments[17].ReturnFloatArgument( );
    float    particle_radius_angstroms     = my_current_job.arguments[18].ReturnFloatArgument( );
    float    phase_shift                   = my_current_job.arguments[19].ReturnFloatArgument( );
    wxString my_symmetry                   = my_current_job.arguments[20].ReturnStringArgument( );
    float    in_plane_angular_step         = my_current_job.arguments[21].ReturnFloatArgument( );
    wxString output_histogram_file         = my_current_job.arguments[22].ReturnStringArgument( );
    int      first_search_position         = my_current_job.arguments[23].ReturnIntegerArgument( );
    int      last_search_position          = my_current_job.arguments[24].ReturnIntegerArgument( );
    float    min_peak_radius               = my_current_job.arguments[25].ReturnFloatArgument( );
    float    phi_opt                       = my_current_job.arguments[26].ReturnFloatArgument( );
    float    theta_opt                     = my_current_job.arguments[27].ReturnFloatArgument( );
    float    psi_opt                       = my_current_job.arguments[28].ReturnFloatArgument( );
    //wxString correlation_avg_output_file   = my_current_job.arguments[29].ReturnStringArgument( );
    //wxString correlation_std_output_file   = my_current_job.arguments[30].ReturnStringArgument( );

    // read in image file (with particle in the center, no randomness in x, y, z, defocus1, defocus2, defocus angle when simulating particles)
    // read in 3d template
    ImageFile input_search_image_file;
    ImageFile input_reconstruction_file;
    input_search_image_file.OpenFile(input_search_images_filename.ToStdString( ), false);
    input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString( ), false);

    Image input_image;
    Image input_reconstruction;
    Image template_reconstruction;
    Image padded_projection;
    Image current_projection;
    Image projection_filter;
    Image padded_reference;

    // array for storing cross-correlations (make sure to store at the optimal pixel and slightly misaligned pixels)
    // TODO histograms should also be extended to a local region?
    double sum_offset_0  = 0.0;
    double sum_offset_1  = 0.0;
    double sum_offset_5  = 0.0;
    double sum_offset_10 = 0.0;
    double sum_offset_50 = 0.0;
    double sos_offset_0  = 0.0;
    double sos_offset_1  = 0.0;
    double sos_offset_5  = 0.0;
    double sos_offset_10 = 0.0;
    double sos_offset_50 = 0.0;

    //Image correlation_pixel_sum_image;
    //Image correlation_pixel_sum_of_squares_image;
    //correlation_pixel_sum_image.Allocate(10, 10, 1, true);
    //correlation_pixel_sum_image.SetToConstant(1.0f);
    //correlation_pixel_sum_of_squares_image.Allocate(10, 10, 1);

    //double* correlation_pixel_sum            = new double[correlation_pixel_sum_image.real_memory_allocated]; // TODO fixme what is a good neighborhood
    //double* correlation_pixel_sum_of_squares = new double[correlation_pixel_sum_of_squares_image.real_memory_allocated];
    //ZeroDoubleArray(correlation_pixel_sum, correlation_pixel_sum_image.real_memory_allocated);
    //ZeroDoubleArray(correlation_pixel_sum_of_squares, correlation_pixel_sum_of_squares_image.real_memory_allocated);

    input_image.ReadSlice(&input_search_image_file, 1);
    input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices( ));
    template_reconstruction.Allocate(input_reconstruction.logical_x_dimension, input_reconstruction.logical_y_dimension, input_reconstruction.logical_z_dimension, true);
    current_projection.Allocate(input_reconstruction_file.ReturnXSize( ), input_reconstruction_file.ReturnXSize( ), false);
    projection_filter.Allocate(input_reconstruction_file.ReturnXSize( ), input_reconstruction_file.ReturnXSize( ), false);
    padded_reference.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    padded_reference.SetToConstant(0.0f);

    if ( padding != 1.0f ) {
        input_reconstruction.Resize(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_y_dimension * padding, input_reconstruction.logical_z_dimension * padding, input_reconstruction.ReturnAverageOfRealValuesOnEdges( ));

        padded_projection.Allocate(input_reconstruction_file.ReturnXSize( ) * padding, input_reconstruction_file.ReturnXSize( ) * padding, false);
    }

    input_reconstruction.ChangePixelSize(&template_reconstruction, 1, 0.001f, true);
    template_reconstruction.ZeroCentralPixel( );
    template_reconstruction.SwapRealSpaceQuadrants( );

    // setup curve
    double sqrt_input_pixels = sqrt((double)(input_image.logical_x_dimension * input_image.logical_y_dimension));

    int   histogram_number_of_points = 512;
    float histogram_min              = -12.5f;
    float histogram_max              = 30.0f;

    float  histogram_step       = (histogram_max - histogram_min) / float(histogram_number_of_points);
    double histogram_min_scaled = histogram_min / sqrt_input_pixels;
    wxPrintf("hist min = %lf\n", histogram_min_scaled);

    double histogram_step_scaled = histogram_step / sqrt_input_pixels;
    wxPrintf("hist step = %lf\n", histogram_step_scaled);
    wxPrintf("hist step = %lf\n", histogram_max / sqrt_input_pixels);

    long* histogram_data_offset_0;
    long* histogram_data_offset_1;
    long* histogram_data_offset_5;
    long* histogram_data_offset_10;
    long* histogram_data_offset_50;
    int   current_bin_offset_0, current_bin_offset_1, current_bin_offset_5, current_bin_offset_10, current_bin_offset_50;

    histogram_data_offset_0  = new long[histogram_number_of_points];
    histogram_data_offset_1  = new long[histogram_number_of_points];
    histogram_data_offset_5  = new long[histogram_number_of_points];
    histogram_data_offset_10 = new long[histogram_number_of_points];
    histogram_data_offset_50 = new long[histogram_number_of_points];

    for ( int counter = 0; counter < histogram_number_of_points; counter++ ) {
        histogram_data_offset_0[counter]  = 0;
        histogram_data_offset_1[counter]  = 0;
        histogram_data_offset_5[counter]  = 0;
        histogram_data_offset_10[counter] = 0;
        histogram_data_offset_50[counter] = 0;
    }
    // set up CTF
    CTF input_ctf;
    input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));
    input_ctf.SetDefocus(defocus1 / pixel_size, defocus2 / pixel_size, deg_2_rad(defocus_angle));
    projection_filter.CalculateCTFImage(input_ctf);

    // preprocess (whitening, normalization)
    Curve whitening_filter;
    Curve number_of_terms;
    whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

    input_image.ReplaceOutliersWithMean(5.0f);
    input_image.ForwardFFT( );
    input_image.SwapRealSpaceQuadrants( );

    input_image.ZeroCentralPixel( );
    input_image.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
    whitening_filter.SquareRoot( );
    whitening_filter.Reciprocal( );
    whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue( ));

    input_image.ApplyCurveFilter(&whitening_filter);
    input_image.ZeroCentralPixel( );
    input_image.DivideByConstant(sqrtf(input_image.ReturnSumOfSquares( )));

    projection_filter.ApplyCurveFilter(&whitening_filter);

    /////// GET optimal location
    // whitened image, 3d template, projection filter filter, optimal pose
    Peak current_peak;
    current_peak = get_optimal_peak(&input_image, &template_reconstruction, &projection_filter, psi_opt, theta_opt, phi_opt, padding);
    int coord_x  = current_peak.x + padded_reference.physical_address_of_box_center_x;
    int coord_y  = current_peak.y + padded_reference.physical_address_of_box_center_y;

    // precalculation find the correct (x,y) location by the correct pose TODO: call another function to get correct coordinates
    // search rotational space and record cross-correlations to histogram
    AnglesAndShifts angles;
    EulerSearch     global_euler_search;
    ParameterMap    parameter_map; // needed for euler search init
    //for (int i = 0; i < 5; i++) {parameter_map[i] = true;}
    parameter_map.SetAllTrue( );
    float variance;
    float current_psi;
    float psi_step;
    float psi_max;
    float psi_start;
    int   number_of_rotations;
    long  total_correlation_positions;
    long  current_correlation_position;
    long  total_correlation_positions_per_thread;
    long  pixel_counter;

    int current_search_position;
    int current_x;
    int current_y;

    // set up search grid
    // in-plane PSI
    psi_step  = in_plane_angular_step;
    psi_start = 0.0f;
    psi_max   = 360.0f;
    // out-of-plane PHI+THETA
    global_euler_search.InitGrid(my_symmetry, angular_step, 0.0f, 0.0f, psi_max, psi_step, psi_start, pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);
    if ( my_symmetry.StartsWith("C") ) // TODO 2x check me - w/o this O symm at least is broken
    {
        if ( global_euler_search.test_mirror == true ) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
        {
            global_euler_search.theta_max = 180.0f;
        }
    }
    global_euler_search.CalculateGridSearchPositions(false);

    total_correlation_positions  = 0;
    current_correlation_position = 0;
    if ( is_running_locally == true ) {
        wxPrintf("running locally...\n");
        first_search_position = 0;
        last_search_position  = global_euler_search.number_of_search_positions - 1;
    }

    for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {
        //loop over each rotation angle

        for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
            total_correlation_positions++;
        }
    }

    number_of_rotations = 0;
    for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
        number_of_rotations++;
    }

    ProgressBar* my_progress;
    if ( is_running_locally == true ) {
        my_progress = new ProgressBar(total_correlation_positions);
    }

    wxPrintf("Searching %i positions on the Euler sphere (first-last: %i-%i)\n", last_search_position - first_search_position, first_search_position, last_search_position);
    wxPrintf("Searching %i rotations per position.\n", number_of_rotations);
    wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions);

    wxPrintf("Performing Search...\n\n");

    // start rotational search
    for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {
        for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
            angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);

            if ( padding != 1.0f ) {
                template_reconstruction.ExtractSlice(padded_projection, angles, 1.0f, false);
                padded_projection.SwapRealSpaceQuadrants( );
                padded_projection.BackwardFFT( );
                padded_projection.ClipInto(&current_projection);
                current_projection.ForwardFFT( );
            }
            else {
                template_reconstruction.ExtractSlice(current_projection, angles, 1.0f, false);
                current_projection.SwapRealSpaceQuadrants( );
            }

            current_projection.MultiplyPixelWise(projection_filter);

            current_projection.BackwardFFT( );

            current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges( ));
            variance = current_projection.ReturnSumOfSquares( ) * current_projection.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels - powf(current_projection.ReturnAverageOfRealValues( ) * current_projection.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels, 2);
            current_projection.DivideByConstant(sqrtf(variance));
            current_projection.ClipIntoLargerRealSpace2D(&padded_reference);

            padded_reference.ForwardFFT( );
            padded_reference.ZeroCentralPixel( );

#ifdef MKL
            // Use the MKL
            vmcMulByConj(padded_reference.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(input_image.complex_values), reinterpret_cast<MKL_Complex8*>(padded_reference.complex_values), reinterpret_cast<MKL_Complex8*>(padded_reference.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
            for ( pixel_counter = 0; pixel_counter < padded_reference.real_memory_allocated / 2; pixel_counter++ ) {
                padded_reference.complex_values[pixel_counter] = conj(padded_reference.complex_values[pixel_counter]) * input_image.complex_values[pixel_counter];
            }
#endif

            padded_reference.BackwardFFT( );
            //padded_reference.QuickAndDirtyWriteSlice("cc_max.mrc", 1);

            // calculate CC @ optimal (x,y) and update histogram
            float current_cc     = padded_reference.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
            current_bin_offset_0 = int(double(current_cc - histogram_min_scaled) / histogram_step_scaled);

            float current_cc_offset_1 = padded_reference.ReturnRealPixelFromPhysicalCoord(coord_x + 1, coord_y + 1, 0);
            current_bin_offset_1      = int(double(current_cc_offset_1 - histogram_min_scaled) / histogram_step_scaled);

            float current_cc_offset_5 = padded_reference.ReturnRealPixelFromPhysicalCoord(coord_x + 5, coord_y + 5, 0);
            current_bin_offset_5      = int(double(current_cc_offset_5 - histogram_min_scaled) / histogram_step_scaled);

            float current_cc_offset_10 = padded_reference.ReturnRealPixelFromPhysicalCoord(coord_x + 10, coord_y + 10, 0);
            current_bin_offset_10      = int(double(current_cc_offset_10 - histogram_min_scaled) / histogram_step_scaled);

            float current_cc_offset_50 = padded_reference.ReturnRealPixelFromPhysicalCoord(coord_x + 20, coord_y + 20, 0);
            current_bin_offset_50      = int(double(current_cc_offset_50 - histogram_min_scaled) / histogram_step_scaled);

            sum_offset_0 += (double)current_cc;
            sum_offset_1 += (double)current_cc_offset_1;
            sum_offset_5 += (double)current_cc_offset_5;
            sum_offset_10 += (double)current_cc_offset_10;
            sum_offset_50 += (double)current_cc_offset_50;
            sos_offset_0 += powf((double)current_cc, 2);
            sos_offset_1 += powf((double)current_cc_offset_1, 2);
            sos_offset_5 += powf((double)current_cc_offset_5, 2);
            sos_offset_10 += powf((double)current_cc_offset_10, 2);
            sos_offset_50 += powf((double)current_cc_offset_50, 2);

            if ( current_bin_offset_0 >= 0 && current_bin_offset_0 <= histogram_number_of_points ) {
                histogram_data_offset_0[current_bin_offset_0] += 1;
            }
            if ( current_bin_offset_1 >= 0 && current_bin_offset_1 <= histogram_number_of_points ) {
                histogram_data_offset_1[current_bin_offset_1] += 1;
            }
            if ( current_bin_offset_5 >= 0 && current_bin_offset_5 <= histogram_number_of_points ) {
                histogram_data_offset_5[current_bin_offset_5] += 1;
            }
            if ( current_bin_offset_10 >= 0 && current_bin_offset_10 <= histogram_number_of_points ) {
                histogram_data_offset_10[current_bin_offset_10] += 1;
            }
            if ( current_bin_offset_50 >= 0 && current_bin_offset_50 <= histogram_number_of_points ) {
                histogram_data_offset_50[current_bin_offset_50] += 1;
            }

            current_correlation_position++;
            if ( is_running_locally == true )
                my_progress->Update(current_correlation_position);
        }
    } // end of rotation search
    long   number_of_result_floats = 0;
    float* pointer_to_histogram_data_offset_0;
    pointer_to_histogram_data_offset_0 = (float*)histogram_data_offset_0;
    float* pointer_to_histogram_data_offset_1;
    pointer_to_histogram_data_offset_1 = (float*)histogram_data_offset_1;
    float* pointer_to_histogram_data_offset_5;
    pointer_to_histogram_data_offset_5 = (float*)histogram_data_offset_5;
    float* pointer_to_histogram_data_offset_10;
    pointer_to_histogram_data_offset_10 = (float*)histogram_data_offset_10;
    float* pointer_to_histogram_data_offset_50;
    pointer_to_histogram_data_offset_50 = (float*)histogram_data_offset_50;

    number_of_result_floats = histogram_number_of_points * sizeof(long) / sizeof(float);
    float* result_offset_0  = new float[number_of_result_floats];
    float* result_offset_1  = new float[number_of_result_floats];
    float* result_offset_5  = new float[number_of_result_floats];
    float* result_offset_10 = new float[number_of_result_floats];
    float* result_offset_50 = new float[number_of_result_floats];

    for ( pixel_counter = 0; pixel_counter < histogram_number_of_points * 2; pixel_counter++ ) {
        result_offset_0[pixel_counter]  = pointer_to_histogram_data_offset_0[pixel_counter];
        result_offset_1[pixel_counter]  = pointer_to_histogram_data_offset_1[pixel_counter];
        result_offset_5[pixel_counter]  = pointer_to_histogram_data_offset_5[pixel_counter];
        result_offset_10[pixel_counter] = pointer_to_histogram_data_offset_10[pixel_counter];
        result_offset_50[pixel_counter] = pointer_to_histogram_data_offset_50[pixel_counter];
    }

    // write out histogram
    float temp_float;
    temp_float = histogram_min + (histogram_step / 2.0f); // start position
    NumericTextFile histogram_file_offset_0(wxString::Format("%s_offset_0.txt", output_histogram_file).ToStdString( ), OPEN_TO_WRITE, 2);
    NumericTextFile histogram_file_offset_1(wxString::Format("%s_offset_1.txt", output_histogram_file).ToStdString( ), OPEN_TO_WRITE, 2);
    NumericTextFile histogram_file_offset_5(wxString::Format("%s_offset_5.txt", output_histogram_file).ToStdString( ), OPEN_TO_WRITE, 2);
    NumericTextFile histogram_file_offset_10(wxString::Format("%s_offset_10.txt", output_histogram_file).ToStdString( ), OPEN_TO_WRITE, 2);
    NumericTextFile histogram_file_offset_50(wxString::Format("%s_offset_50.txt", output_histogram_file).ToStdString( ), OPEN_TO_WRITE, 2);

    histogram_file_offset_0.WriteCommentLine("SNR, histogram");
    histogram_file_offset_1.WriteCommentLine("SNR, histogram");
    histogram_file_offset_5.WriteCommentLine("SNR, histogram");
    histogram_file_offset_10.WriteCommentLine("SNR, histogram");
    histogram_file_offset_50.WriteCommentLine("SNR, histogram");

    double temp_double_array[2];

    for ( int line_counter = 0; line_counter < histogram_number_of_points; line_counter++ ) {
        temp_double_array[0] = temp_float + histogram_step * float(line_counter);
        temp_double_array[1] = histogram_data_offset_0[line_counter];
        histogram_file_offset_0.WriteLine(temp_double_array);
        temp_double_array[1] = histogram_data_offset_1[line_counter];
        histogram_file_offset_1.WriteLine(temp_double_array);
        temp_double_array[1] = histogram_data_offset_5[line_counter];
        histogram_file_offset_5.WriteLine(temp_double_array);
        temp_double_array[1] = histogram_data_offset_10[line_counter];
        histogram_file_offset_10.WriteLine(temp_double_array);
        temp_double_array[1] = histogram_data_offset_50[line_counter];
        histogram_file_offset_50.WriteLine(temp_double_array);
    }
    histogram_file_offset_0.Close( );
    histogram_file_offset_1.Close( );
    histogram_file_offset_5.Close( );
    histogram_file_offset_10.Close( );
    histogram_file_offset_50.Close( );

    sum_offset_0 /= float(total_correlation_positions);
    float var_offset_0 = sos_offset_0 / float(total_correlation_positions) - powf(sum_offset_0, 2);
    if ( var_offset_0 > 0.0f )
        var_offset_0 = sqrtf(var_offset_0) * (float)sqrt_input_pixels;
    else
        var_offset_0 = 0.0f;
    sum_offset_0 *= (float)sqrt_input_pixels;

    sum_offset_1 /= float(total_correlation_positions);
    float var_offset_1 = sos_offset_1 / float(total_correlation_positions) - powf(sum_offset_1, 2);
    if ( var_offset_1 > 0.0f )
        var_offset_1 = sqrtf(var_offset_1) * (float)sqrt_input_pixels;
    else
        var_offset_1 = 0.0f;
    sum_offset_1 *= (float)sqrt_input_pixels;

    sum_offset_5 /= float(total_correlation_positions);
    float var_offset_5 = sos_offset_5 / float(total_correlation_positions) - powf(sum_offset_5, 2);
    if ( var_offset_5 > 0.0f )
        var_offset_5 = sqrtf(var_offset_5) * (float)sqrt_input_pixels;
    else
        var_offset_5 = 0.0f;
    sum_offset_5 *= (float)sqrt_input_pixels;

    sum_offset_10 /= float(total_correlation_positions);
    float var_offset_10 = sos_offset_10 / float(total_correlation_positions) - powf(sum_offset_10, 2);
    if ( var_offset_10 > 0.0f )
        var_offset_10 = sqrtf(var_offset_10) * (float)sqrt_input_pixels;
    else
        var_offset_10 = 0.0f;
    sum_offset_10 *= (float)sqrt_input_pixels;

    sum_offset_50 /= float(total_correlation_positions);
    float var_offset_50 = sos_offset_50 / float(total_correlation_positions) - powf(sum_offset_50, 2);
    if ( var_offset_50 > 0.0f ) {
        var_offset_50 = sqrtf(var_offset_50) * (float)sqrt_input_pixels;
    }
    else
        var_offset_50 = 0.0f;
    sum_offset_50 *= (float)sqrt_input_pixels;

    wxPrintf("offset = 0: %f %f\n", sum_offset_0, var_offset_0);
    wxPrintf("offset = 1: %f %f\n", sum_offset_1, var_offset_1);
    wxPrintf("offset = 5: %f %f\n", sum_offset_5, var_offset_5);
    wxPrintf("offset = 10: %f %f\n", sum_offset_10, var_offset_10);
    wxPrintf("offset = 50: %f %f\n", sum_offset_50, var_offset_50);

    return true;
}
