 double perc_diff_hue = ((hue - mean_hue) / mean_hue;
 double perc_diff_sat = ((saturation[1] - mean_saturation) / mean_saturation) * 100;
 double perc_diff_val = ((mean_value - value[2]) / mean_value[2]) * 100;

 if (perc_diff_sat > 12.0 || perc_diff_val > 25.0) {
	if ((perc_diff_hue > 3.25)) {
		if (perc_diff_val < 10.0) {
			lesion is red
		}
		else if ((perc_diff_sat < 25.0) && (perc_diff_val < 30.0)) {
			lesion is red
		}
		else {
			lesion is dark
		}
	}
	else {
		if (perc_diff_val < 5.0) {
			lesion is red
		}
		else {
			lesion is dark
		}
	}
 }
 else {
	 lesion is too similar to the skin
 }


