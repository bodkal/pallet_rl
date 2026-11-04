# One carton, in real cm, in the pallet frame: origin at the pallet corner
# where the grid starts, x along the pallet length, y along its width,
# z up from the deck.
#
# Placed boxes: all fields.  Station boxes: only the sides and the type.
float32 x_cm        # centre along x
float32 y_cm        # centre along y
float32 z_cm        # bottom face
float32 length_cm   # side along x, as the box stands
float32 width_cm    # side along y
float32 height_cm
uint8 type          # 0 blue, 1 white, 2 brown
