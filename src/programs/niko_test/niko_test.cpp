#include "../../core/core_headers.h"

class
        NikoTestApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(NikoTestApp)

// override the DoInteractiveUserInput

void NikoTestApp::DoInteractiveUserInput( ) {
}

// override the do calculation method which will be what is actually run..

bool NikoTestApp::DoCalculation( ) {
    Image test_image;
    test_image.QuickAndDirtyReadSlice("./s_THP1_24hbr_g1_niceview_00003_20.0_Mar18_07.51.09_13_0.mrc", 1);
    wxPrintf("image size %i %i\n", test_image.logical_x_dimension, test_image.logical_y_dimension);

    Image cropped_image;
    cropped_image.Allocate(1700, 1400, true);
    test_image.ClipInto(&cropped_image);

    cropped_image.ForwardFFT( );
    cropped_image.GaussianLowPassFilter(0.1);
    cropped_image.QuickAndDirtyWriteSlice("cropped.mrc", 1, true);
    return true;
}
