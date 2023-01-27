package kiv.bp.bp_android.engines;

import android.graphics.Bitmap;

import org.pytorch.IValue;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.nio.FloatBuffer;

/**
 * The type Neural engine.
 */
public class NeuralEngine {


    /**
     * The pytorch Module.
     */
    private final Module module;


    private Bitmap processedPicture;

    /**
     * Instantiates a new Neural engine.
     *
     * @param modulePath the module path
     */
    public NeuralEngine(String  modulePath) {
        this.module =  Module.load(modulePath);
    }

    /**
     * Predict picture
     *
     * @return the float [ ] scores in softmax
     */
    public float[]  predict() {
        // This is normalizing the image.
        float[] normMeanRGB = new float[] {0.485f, 0.456f, 0.406f};
        float[] normStdRGB = new float[] {0.229f, 0.224f, 0.225f};

        // Creating an array of integers with the size of the width and height of the image.
        final int[] pixels = new int[processedPicture.getWidth() * processedPicture.getHeight()];
        // This is getting the pixels from the bitmap and putting them into a FloatBuffer.
        processedPicture.getPixels(pixels, 0, processedPicture.getWidth(), 0, 0, processedPicture.getWidth(), processedPicture.getHeight());
        FloatBuffer outBuffer = Tensor.allocateFloatBuffer(3 * processedPicture.getWidth() * processedPicture.getHeight());
        float r;
        float g;
        float bb;
        int i = 0;
        // This is getting the pixels from the bitmap and putting them into a FloatBuffer.
        for (int c : pixels) {
            r = ((c >> 16) & 0xff) / 255.0f;
            g = ((c >> 8) & 0xff) / 255.0f;
            bb = ((c) & 0xff) / 255.0f;
            outBuffer.put(3 * i, (r - normMeanRGB[0]) / normStdRGB[0]);
            outBuffer.put( 3 * i + 1, (g - normMeanRGB[1]) / normStdRGB[1]);
            outBuffer.put( 3 * i + 2, (bb - normMeanRGB[2]) / normStdRGB[2]);
            i++;
        }

        // preparing input tensor
        final Tensor inputTensor = Tensor.fromBlob(outBuffer, new long[] {1, 3, processedPicture.getWidth(), processedPicture.getHeight()}, MemoryFormat.CHANNELS_LAST);

        // running the model
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // getting tensor content as java array of floats
         return outputTensor.getDataAsFloatArray();
    }


    /**
     * This function sets the processedPicture variable to the processedPicture parameter.
     *
     * @param processedPicture The bitmap that will be displayed in the ImageView.
     */
    public void setProcessedPicture(Bitmap processedPicture) {
        this.processedPicture = processedPicture;
    }

}
