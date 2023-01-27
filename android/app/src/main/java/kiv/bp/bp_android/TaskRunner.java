package kiv.bp.bp_android;

import android.os.Handler;
import android.os.Looper;

import java.util.concurrent.Callable;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;


/**
 * A class that runs tasks.
 */
public class TaskRunner {
    // Creating a single thread executor.
    private final Executor executor = Executors.newSingleThreadExecutor();
    // Creating a handler that is tied to the main thread.
    private final Handler handler = new Handler(Looper.getMainLooper());



    public <SOmeType> void executeAsync(Callable<SOmeType> callable, Callback<SOmeType> callback) {
        executor.execute(() -> {
            try {
                final SOmeType result = callable.call();
                handler.post(() -> {
                    callback.onComplete(result);
                });
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }
}
