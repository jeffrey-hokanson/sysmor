def _get_dimensions(zs):
	# Determine dimensions	
	p = 0
	m = 0
	for i, j in zs:
		p = max(p, i)
		m = max(m, j)
	p += 1
	m += 1
	return p, m
