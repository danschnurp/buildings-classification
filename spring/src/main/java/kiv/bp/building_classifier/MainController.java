package kiv.bp.building_classifier;

import ai.onnxruntime.OrtException;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.PostConstruct;
import javax.imageio.IIOException;
import javax.imageio.ImageIO;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import java.io.File;
import java.io.IOException;


/**
 * The type Main controller.
 */
@Controller
public class MainController {

    /**
     * The Engine.
     */
    ClassificationEngine engine;

    /**
     * Init.
     */
    @PostConstruct
    public void init() {
        engine = new ClassificationEngine();
        try {
                        engine.init((this.getClass().getResourceAsStream("/WEB-INF/classes/model_densenet121_2022-03-06.onnx")));
//            engine.init(new File("src/main/resources/model_densenet121_2022-03-06.onnx").getPath());
        } catch (OrtException | IOException e) {
            System.err.println("WRONG PATH TO MODEL!");
            e.printStackTrace();
        }

    }


    /**
     * Main poster string.
     *
     * @param model the model
     * @return the string
     */
    @GetMapping(value = "/")
    public String mainPoster(Model model) {
        model.addAttribute("empty", "true");
        model.addAttribute("description", "");
        model.addAttribute("class_name1", "");
        return "main";
    }

    /**
     * Main poster result string.
     *
     * @param model   the model
     * @param request the request
     * @param file    the file
     * @return the string
     */
    @PostMapping(value = "/result")
    public String mainPosterResult(Model model, HttpServletRequest request, @RequestParam(value = "file") MultipartFile file) {
        try {
            if (isJavaScriptDisabled(request)) {
                model.addAttribute("empty", "blocked");
                model.addAttribute("description", "");
                model.addAttribute("class_name1", "Cookies are blocked or JavaScript is disabled...");
                return "main";
            }

            engine.b = ImageIO.read(file.getInputStream());

            if (engine.b == null) {
                model.addAttribute("empty", "false");
                model.addAttribute("description", "");
                model.addAttribute("class_name1", "Bad input...");
                return "main";
            }


            String[] result = engine.onnxPrepare();
            if (result.length == 1)
                throw new Exception();


            model.addAttribute("empty", "false");
            model.addAttribute("description", "Predicted architectonic style: ");
            model.addAttribute("class_name1", result[0]);
            model.addAttribute("class_name2", result[1]);
            model.addAttribute("class_name3", result[2]);
            return "main";
        }
        catch (IIOException e) {
            e.printStackTrace();
            model.addAttribute("empty", "false");
            model.addAttribute("description", "Pictures must have 24 bit depth.");
            model.addAttribute("class_name1", "Unsupported input... ");
            return "main";
        }
        catch (Exception e) {
            e.printStackTrace();
            model.addAttribute("empty", "false");
            model.addAttribute("description", "");
            model.addAttribute("class_name1", "Corrupted input...");
            return "main";
        }

    }

    private boolean isJavaScriptDisabled(HttpServletRequest request)
    {
        boolean isJavaScriptDisabled = true;
        Cookie[] cookies = request.getCookies();

        if (cookies != null)
        {
            for (int i = 0; i < cookies.length; i++)
            {
                if ("JavaScriptEnabledCheck".equalsIgnoreCase(cookies[i].getName()))
                {
                    isJavaScriptDisabled = false;
                    break;
                }
            }
        }

        return isJavaScriptDisabled;
    }


}
