package kiv.bp.building_classifier;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import com.google.common.io.ByteStreams;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;

/**
 * The type Classification engine.
 */
public class ClassificationEngine {


    /**
     * The BufferedImage.
     */
    BufferedImage b = null;
    /**
     * The Session.
     */
    OrtSession session;
    /**
     * The Env.
     */
    OrtEnvironment env;

    /**
     * Init.
     *
     * @param path the path
     * @throws OrtException the ort exception
     * @throws IOException  the io exception
     */
// String
    public void init(InputStream path) throws OrtException, IOException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new SessionOptions();
        opts.setOptimizationLevel(OptLevel.BASIC_OPT);
        if (path == null)
            throw new IOException();
        byte[] bytes = ByteStreams.toByteArray(path);
        session = env.createSession(bytes, opts);
//        session = env.createSession(path, opts);
    }

    private BufferedImage resize(BufferedImage img, int newW, int newH) {
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);

        Graphics2D g2d = dimg.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return dimg;
    }

    private void resize() {
        b = resize(b, 500, 500);
    }

    /**
     * Onnx prepare string [ ] of results.
     *
     * @return the string [ ]
     * @throws OrtException the ort exception
     */
    public String[]  onnxPrepare() throws OrtException {
        resize();



                // Normalizing the image.
                float[] normMeanRGB = new float[] {0.485f, 0.456f, 0.406f};
                float[] normStdRGB = new float[] {0.229f, 0.224f, 0.225f};

                // Creating a 4D array of floats.
                final float[][][][] pixels = new float[1][3][500][500];
                Raster raster = b.getRaster();
                float r;
                float g;
                float bb;
                int sample =0;
                // Normalizing the image.
                for (int i = 0; i < b.getHeight(); i++) {
                    for (int j = 0; j < b.getWidth(); j++) {
                        sample = raster.getSample(j,i,0);
                        r = ((sample >> 16) & 0xff) / 255.0f;
                        g = ((sample >> 8) & 0xff) / 255.0f;
                        bb = ((sample) & 0xff) / 255.0f;
                        pixels[0][0][i][j] = ((r - normMeanRGB[0]) / normStdRGB[0]);
                        pixels[0][1][i][j] = ((g - normMeanRGB[1]) / normStdRGB[1]);
                        pixels[0][2][i][j] = ((bb - normMeanRGB[2]) / normStdRGB[2]);
                    }
                }


                // Creating a tensor from the pixels array.
                OnnxTensor tensor = OnnxTensor.createTensor(env, pixels);
        HashMap<String, OnnxTensor> onnxTensorMap  = new HashMap<>();
        onnxTensorMap.put("input", tensor);
                // Running the session with the input tensor.
                try (Result results = session.run(onnxTensorMap)) {
                    float[][] outputProbs = (float[][]) results.get(0).getValue();
                    return pred(outputProbs[0]);
                }

                 catch (OrtException e) {
                            e.printStackTrace();
                     return new String[]{"error"};
                        }
    }

    private String[] pred(float[] probabilities) {
        int[] mostIDs = {-1, -1, -1};
        float[] mostScores = {
                -Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE
        };
        // A list of all the classes that the model can predict.
        String[] classes = new String[]{"Art Nouveau", "Architecture of ancient China / Japan / Korea", "Baroque",
                "Brutalist Architecture", "Cubism", "Functionalism", "Gothic Architecture", "Islamic Architecture",
                "Renaissance",
                "Romanesque"};

        // searching for the index with maximum score
        for (int j = 0; j < classes.length; j++) {
            for (int k = 0; k < mostScores.length; k++) {
                if (probabilities[j] > mostScores[k]) {
                    mostScores[k] = probabilities[j];
                    mostIDs[k] = j;
                    break;
                }
            }
        }
        // Returning the top 3 predictions.
        return new String[]{"1. predicted: " + classes[mostIDs[0]] + "\n" ,
                "2. predicted: " + classes[mostIDs[1]] + "\n" ,
                "3. predicted: " + classes[mostIDs[2]] + "\n"};
    }

}


