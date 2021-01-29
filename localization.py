import cv2 as cv

# INPUT:
# room = room of the current frame
def localize(room):
    # load of map and room images
    # template = room image
    map = cv.imread('localization/map.png', 0)
    temp_path = 'localization/' + str(room) + '.png'
    template = cv.imread(temp_path, 0)
    w, h = template.shape[::-1]

    # Correlation Coefficient mode for template matching
    meth = 'cv.TM_CCOEFF'
    method = eval(meth)

    # Template Matching
    res = cv.matchTemplate(map, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w + 10, top_left[1] + h + 10)

    # Draw a rectangle around the matched room in the map
    cv.rectangle(map, top_left, bottom_right, 0, 20)
    map = cv.resize(src=map, dsize=(0, 0), dst=None, fx=0.5, fy=0.5)
    cv.imshow('template', map)
