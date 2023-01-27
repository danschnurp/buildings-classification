package kiv.bp.bp_android;

// A generic interface that is used to pass a callback function to the `getData` function.
public interface Callback<R> {
    void onComplete(R result);
}
