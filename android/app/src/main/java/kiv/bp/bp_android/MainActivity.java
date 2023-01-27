package kiv.bp.bp_android;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import kiv.bp.bp_android.engines.NeuralEngine;

/**
 * The type Main activity.
 *
 * @author Daniel Schnurpfeil
 */
public class MainActivity extends AppCompatActivity {

    /**
     * The Image view.
     */
    private ImageView imageView;
    /**
     * The B.
     */
    private Bitmap processedPicture = null;
    /**
     * The S.
     */
    private String resultString;
    /**
     * The S 2.
     */
    private String secondResultString;
    /**
     * The Info.
     */
    private AlertDialog info;
    /**
     * The TextView.
     */
    private TextView resultTextView;
    /**
     * The TextView2.
     */
    private TextView secondResultTextView;
    /**
     * The Button takePicture.
     */
    private Button buttonTakePicture;
    /**
     * The Button loadPicture.
     */
    private Button buttonLoadPicture;
    /**
     * The ProgressBar.
     */
    private ProgressBar progressBar;

    /**
     * The Alert.
     */
    private AlertDialog permissionsAlert;

    private TaskRunner taskRunner;

    /**
     * The Neural nett name.
     */
    private final String neuralNettName = "model_mobile.pt";

    private NeuralEngine engine;

    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        verifyStoragePermissions(this);
        setContentView(R.layout.activity_main);
        resultTextView = findViewById(R.id.textView);
        secondResultTextView = findViewById(R.id.textView2);
        progressBar = findViewById(R.id.progressBar);
        progressBar.getIndeterminateDrawable().setColorFilter(Color.parseColor("#EFC941"),
                PorterDuff.Mode.SRC_IN);
        resultTextView.setText(R.string.promt);
        imageView = findViewById(R.id.imageView);
        buttonTakePicture = findViewById(R.id.button);
        buttonLoadPicture = findViewById(R.id.button2);
        Button buttonInfo = findViewById(R.id.buttonInfo);
        taskRunner = new TaskRunner() ;
        createAlert();
        buttonTakePicture.setOnClickListener(view -> takePicture());
        buttonLoadPicture.setOnClickListener(view -> loadPicture());
        makeInfo();
        buttonInfo.setOnClickListener(view -> loadInfo());
        try {
            // loading serialized torchscript module from packaged into app android asset model_mobile.pt,
            engine = new NeuralEngine(assetFilePath(this, neuralNettName));
        } catch (IOException e) {
            Log.e("fff", "Error reading assets", e);
            finish();
        }
    }

    /**
     * Load picture.
     */
    private void loadPicture() {
        Intent i = new Intent(
                Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(i, 1);
    }


    /**
     * Take picture.
     */
    private void takePicture() {
        Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(i, 2);
    }


    @SuppressLint("SetTextI18n")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent dataI) {
        super.onActivityResult(requestCode, resultCode, dataI);
        try {
            if (resultCode != RESULT_CANCELED) {
                if (requestCode == 1 && resultCode == RESULT_OK && null != dataI) {
                    Uri selectedImage = dataI.getData();
                    String[] filePathColumn = {MediaStore.Images.Media.DATA};

                    Cursor cursor = getContentResolver().query(selectedImage,
                            filePathColumn, null, null, null);
                    cursor.moveToFirst();

                    // getting the path of the image that was selected by the user
                    int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                    String picturePath = cursor.getString(columnIndex);
                    cursor.close();
                    processedPicture = loadImage(picturePath);

                } else if (requestCode == 2 && resultCode == RESULT_OK) {
                    processedPicture = (Bitmap) dataI.getExtras().get("data");
                }
                showAndProcessImage();
            }
        } catch (Exception e) {
            permissionsAlert.show();
        }
    }


    /**
     * Make info.
     */
    private void makeInfo() {
        final String[] values = getResources().getStringArray(R.array.styles);
        StringBuilder styles = new StringBuilder();
        for (String value : values) {
            styles.append(value).append("\n");
        }
        // Creating a new AlertDialog.Builder object.
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder
                .setTitle("INFO")
                .setIcon(android.R.drawable.ic_menu_info_details)
                .setMessage(styles +
                        "\n\n" +
                        getString(R.string.neco) + "\n\n" +
                        " 2022 (c) Daniel Schnurpfeil")
                .setPositiveButton("OK", (dialog, which) -> dialog.dismiss());
        info = builder.create();

    }

    /**
     * Load info.
     */
    private void loadInfo() {

        info.show();

    }

    /**
     * Create alert.
     */
    private void createAlert() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder
                .setTitle(R.string.permisions)
                .setIcon(android.R.drawable.ic_dialog_alert)
                .setPositiveButton("OK", (dialog, which) -> dialog.dismiss());
        permissionsAlert = builder.create();
    }

    /**
     * Show and process image.
     */
    private void showAndProcessImage() {
        // resizing the image to the size of the imageView
        imageView.setImageBitmap(resize(processedPicture, imageView.getMeasuredWidth(),
                imageView.getMeasuredHeight()));
        // Scaling the image to 500x500 pixels.
        processedPicture = Bitmap.createScaledBitmap(processedPicture, 500, 500, true);

        engine.setProcessedPicture(processedPicture);

        resultTextView.setTextSize(25);
        progressBar.setVisibility(View.VISIBLE);
        resultTextView.setText(R.string.working);
        secondResultTextView.setText("");
        buttonTakePicture.setOnClickListener(null);
        buttonLoadPicture.setOnClickListener(null);

        // Executing the engine asynchronously and passing the result to the NeuroResult class.
        taskRunner.executeAsync(new NeuroResult(engine), (data) -> {

            classifyImage(data);

            TextView t = findViewById(R.id.textView);
            t.setText(resultString);
            secondResultTextView.setText(secondResultString);
            t.setTextSize(25);
            progressBar.setVisibility(View.GONE);

            buttonTakePicture.setOnClickListener(view -> takePicture());
            buttonLoadPicture.setOnClickListener(view -> loadPicture());

                });



    }

    /**
     * Resize bitmap.
     *
     * @param image     the image
     * @param maxWidth  the max width
     * @param maxHeight the max height
     * @return the bitmap
     */
    private static Bitmap resize(Bitmap image, int maxWidth, int maxHeight) {
        if (maxHeight > 0 && maxWidth > 0) {
            int width = image.getWidth();
            int height = image.getHeight();
            float ratioBitmap = (float) width / (float) height;
            float ratioMax = (float) maxWidth / (float) maxHeight;

            int finalWidth = maxWidth;
            int finalHeight = maxHeight;
            if (ratioMax > ratioBitmap) {
                finalWidth = (int) ((float)maxHeight * ratioBitmap);
            } else {
                finalHeight = (int) ((float)maxWidth / ratioBitmap);
            }
            image = Bitmap.createScaledBitmap(image, finalWidth, finalHeight, true);
        }
        return image;
    }



    /**
     * Checks if the app has permission to write to device storage
     * <p>
     * If the app does not has permission then the user will be prompted to grant permissions
     *
     * @param activity main activity
     */
    private void verifyStoragePermissions(Activity activity) {
        // Check if we have write permission
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);

        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    activity,
                    new String[]{
                            Manifest.permission.READ_EXTERNAL_STORAGE,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    },
                    1 // 1 = true = need request
            );
        }
    }


    /**
     * Load image bitmap.
     *
     * @param picturePath the picture path
     * @return the bitmap
     */
    private Bitmap loadImage(String  picturePath) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(picturePath, options);
        options.inSampleSize = calculateInSampleSize(options);
        // Decode bitmap with inSampleSize set
        options.inJustDecodeBounds = false;
        return BitmapFactory.decodeFile(picturePath, options);
    }

    /**
     * Calculate in sample size int.
     *
     * @param options the options
     * @return the int
     */
    private int calculateInSampleSize(BitmapFactory.Options options) {
        int defaultSize = 500;
        // Raw height and width of image
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;

        if (height > defaultSize || width > defaultSize) {

            final int halfHeight = height / 2;
            final int halfWidth = width / 2;

            // Calculate the largest inSampleSize value that is a power of 2 and keeps both
            // height and width larger than the requested height and width.
            while ((halfHeight / inSampleSize) >= defaultSize
                    && (halfWidth / inSampleSize) >= defaultSize) {
                inSampleSize *= 2;
            }
        }
        return inSampleSize;
    }

    /**
     * Classify image.
     */
    private void classifyImage(float[] scores) {

        String[] _CLASSES = getResources().getStringArray(R.array.results);

        float[] mostScores = {
                -Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE
        };
        int[] mostIDs = {-1, -1, -1};

        // searching for the index with maximum score
        for (int j = 0; j < _CLASSES.length; j++) {
            for (int k = 0; k < mostIDs.length; k++) {
                if (scores[j] > mostScores[k]) {
                    mostScores[k] = scores[j];
                    mostIDs[k] = j;
                    break;
                }
            }
        }
        resultString = "1. " + getString(R.string.predicted) + " \n" + _CLASSES[mostIDs[0]] + "\n";
        secondResultString = "\n 2. " + getString(R.string.predicted) + " \n" + _CLASSES[mostIDs[1]] + "\n" +
                "\n" +
                "3. " + getString(R.string.predicted) + " \n" + _CLASSES[mostIDs[2]] + "\n \n";
    }

    /**
     * Copies specified asset to the file in /files app directory
     *
     * @param context   the context
     * @param assetName the asset name
     * @return absolute file path
     * @throws IOException the io exception
     */
    private String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}
