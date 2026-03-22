from spectral_synthesis import *


class ReservoirSource(Source):
    def __init__(self, source: Source, reservoir_size: int):
        self.reservoir: numpy.ndarray = source.photons(reservoir_size)

    def photons(self, sample_size: int) -> numpy.ndarray:
        return numpy.random.choice(self.reservoir, size=sample_size)


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-x', '--skip-simulation', action='store_true')
    parser.add_argument('-l', '--load-checkpoint')
    parser.add_argument('-e', '--escape-events', action='store_true')
    args = parser.parse_args()

    import seaborn

    # Setup default figure size and use LaTeX for text formatting
    seaborn.set_theme(style='darkgrid',
                      palette='Spectral',
                      rc={
                          'axes.facecolor': '.8',
                          'figure.figsize': [12, 8],
                          'figure.dpi': 144,
                          'text.usetex': True
                          })
    pyplot.rc('figure', figsize=[12, 8], dpi=144)
    pyplot.rc('text', usetex=True)

    # Load run configuration
    with open(f'{args.prefix}-config.json', 'r') as fyle:
        run_config = json.load(fyle)

    density = run_config["density"] # Expect high particle number densities ~ O(10^30 m^-3)
    temperature = run_config["temperature"] # Expect high temperatures ~ O(10^6 K)

    # Higher the thickness photons escape less often and get frequency shifted or absorbed
    # Only escaped photons are observed; step count may have to be increased to allow photons to escape
    thickness = run_config["thickness"]

    test_wavelength = run_config["test_wavelength"] # Expect small wavelengths ~ Gamma/X-ray/UV

    # Allow re-emission of absorbed photons as thermal photons following Planck's law distribution
    re_emit = run_config["re_emit"]

    photons = run_config["photons"]
    steps = run_config["steps"]
    checkpoints = run_config["checkpoints"]

    if not args.skip_simulation:
        # Begin 1D Random Walk and Monte Carlo integration

        # Setup a simple isotropic atmosphere with ground state Hydrogen atoms, protons and electrons
        # Expect highly ionising temperatures to nullify Hydrogen atom number densities
        electron = Particle('e', 0, 1, -1)
        thomson_scattering = ThomsonScattering(electron, electron)
        free_free_absorption = FreeFreeAbsorption(electron, electron)
        transitions = set([thomson_scattering, free_free_absorption])

        hydrogen_atom = Particle('H', 1, 1, 0)
        hydrogen_ion = Particle('H+', 1, 1, 1)
        particles = set([hydrogen_atom, hydrogen_ion])
        transitions.add(BoundFreeAbsorption(hydrogen_atom, hydrogen_ion))

        atmosphere = Atmosphere(electron, particles, transitions, {hydrogen_atom: 1.},
                                thickness, ZeroGradient(), ZeroGradient(), core_density=density, core_temperature=temperature)
        # Extract a cell from isotropic atmosphere to get the number densities
        cell = atmosphere.cells(1.)[0]

        # Thomson scattering opacity is constant given the atmosphere is isotropic
        # Extract the value to avoid recalculation on every step
        thomson_scattering_opacity = thomson_scattering.opacity(0., cell)

        # Calculate the Compton wavelength shift
        #
        #   delta_lambda = (1 - cos(theta)) * h / (m_e * c)
        #
        # In 1D, scatter angle (theta) is either 0 (no scatter) or pi (direction reversal)
        delta_wavelength_compton = 2 * scipy.constants.h / (scipy.constants.m_e * scipy.constants.c)

        # Calculate standard deviation for electron thermal velocity
        # based on 1D Maxwell-Boltzmann distribution
        #
        #   sigma = sqrt(k * T / m_e)
        #
        # Thermal velocity of electron would result in Doppler shift of photon wavelength
        # during the scattering event
        electron_thermal_velocity_deviation = numpy.sqrt(scipy.constants.k * temperature / scipy.constants.m_e)

        # Keep a reservoir of photons following Planck's law distribution
        # to re-emit after free-free absorption
        black_body_source = ReservoirSource(BlackBodySource(temperature), 1000000)

        # Start photons at test wavelength
        wavelengths = numpy.full(photons, test_wavelength)

        # Initialise photons at origin in positive direction
        allowed_directions = [-1, 1]
        directions = numpy.ones_like(wavelengths)
        positions = numpy.zeros_like(wavelengths)

        # Track photon states
        absorbed = numpy.full_like(wavelengths, False, dtype=bool)
        escaped = numpy.full_like(wavelengths, False, dtype=bool)
        re_emission_events = numpy.zeros((1,), dtype=int)
        escape_events_at_step = numpy.zeros((steps,), dtype=int)

        # Begin random walk
        for step in range(steps):
            # Checkpoint at chosen steps to observe evolution of frequency shift due to scattering
            if step in checkpoints:
                numpy.savez(f'{args.prefix}-checkpoint-{step}.npz',
                            wavelengths=wavelengths,
                            directions=directions,
                            positions=positions,
                            absorbed=absorbed,
                            escaped=escaped,
                            re_emission_events=re_emission_events,
                            escape_events_at_step=escape_events_at_step)
            
            # Calculate opacity of free-free absorption and net opacity for interaction with photon
            free_free_absorption_opacity = free_free_absorption.opacity(wavelengths, cell)
            total_opacity = thomson_scattering_opacity + free_free_absorption_opacity

            # Sample an optical depth from exponential distribution
            # and calculate physical distance for photon to travel for next interaction
            # Walk the photon to the interaction by this distance in direction of travel
            #
            #   distance_travelled = optical_depth * mean_free_path
            #
            #   mean_free_path = 1 / opacity
            #
            # See
            #   Carroll, B. W., & Ostlie, D. A. 2018, An introduction to modern astrophysics, p. 238-244
            #   (Second edition; Cambridge: Cambridge University Press)
            #
            optical_depths = numpy.random.exponential(size=wavelengths.size)
            positions += directions * optical_depths / total_opacity

            # Check if photon escaped
            escaped_in_step = ~escaped & (positions > thickness)
            escaped |= escaped_in_step
            escape_events_at_step[step] += numpy.count_nonzero(escaped_in_step)

            # If photon hasn't escaped, sample probability that it is absorbed
            absorbed_in_step = ~escaped & (numpy.random.uniform(0, 1, size=wavelengths.size) < free_free_absorption_opacity)

            # If re-emission is not allowed subsequent effects are ignored for these photons
            if not re_emit:
                absorbed |= absorbed_in_step
            else:
                # Otherwise re-emit following Planck's law
                re_emit_sample_size = numpy.count_nonzero(absorbed_in_step)
                numpy.putmask(wavelengths, absorbed_in_step, black_body_source.photons(re_emit_sample_size))
                re_emission_events += re_emit_sample_size

            # Escaped or permanently absorbed photons ignore the effects applied hereafter
            # Photons absorbed in step can change direction and be affected by subsequent steps
            non_interactive = escaped | absorbed_in_step | absorbed

            # Sample new direction of travel
            updated_directions = numpy.random.choice(allowed_directions, size=wavelengths.size)
            # Dot product before and after vectors to check for change in direction
            direction_change = ((directions * updated_directions) < 0)

            # If photon escaped or is absorbed or has no change in direction, it is not Compton scattered
            # If Compton scattered, shift wavelength
            wavelengths += numpy.where(non_interactive | ~direction_change, 0., delta_wavelength_compton)

            # Update the direction vectors
            directions = updated_directions

            # Sample thermal velocities of electrons from Gaussian distribution
            # and calculate Doppler shift
            electron_thermal_velocities = numpy.random.normal(0, electron_thermal_velocity_deviation, size=wavelengths.size)
            doppler_shift_factor = 1 / (1 + electron_thermal_velocities / scipy.constants.c)

            # Apply Doppler shift if photon is not absorbed or escaped
            # Even if not Compton scattered, wavelength is Doppler shifted due to Thomson scattering
            # since the photon has been walked to an interaction point one of these events must occur
            wavelengths *= numpy.where(non_interactive, 1., doppler_shift_factor)
        
        numpy.savez(f'{args.prefix}-checkpoint-{steps}.npz',
                    wavelengths=wavelengths,
                    directions=directions,
                    positions=positions,
                    absorbed=absorbed,
                    escaped=escaped,
                    re_emission_events=re_emission_events,
                    escape_events_at_step=escape_events_at_step)
    else:
        arrays = numpy.load(f'{args.prefix}-checkpoint-{args.load_checkpoint or steps}.npz')
        wavelengths = arrays['wavelengths']
        directions = arrays['directions']
        positions = arrays['positions']
        absorbed = arrays['absorbed']
        escaped = arrays['escaped']
        re_emission_events = arrays['re_emission_events']
        escape_events_at_step = arrays['escape_events_at_step']

    escape_events = numpy.count_nonzero(escaped)
    LOG.info(f'Escaped: {escape_events/wavelengths.size:.5f}')
    LOG.info(f'Mean steps to escape: {numpy.sum(numpy.arange(steps) * escape_events_at_step) / escape_events:.2f}')
    LOG.info(f'Absorbed: {numpy.count_nonzero(absorbed)/wavelengths.size:.5f}')
    LOG.info(f'Re-emissions: {numpy.sum(re_emission_events)}')

    # Calculate spectrum of escaped wavelengths
    intensity, bin_edges = numpy.histogram(wavelengths[escaped], bins=25000)
    output_wavelengths = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot the spectrum
    compact_test_wavelength = test_wavelength * 1e12
    compact_test_wavelength_unit = 'pm'
    if compact_test_wavelength > 1000:
        compact_test_wavelength = test_wavelength * 1e9
        compact_test_wavelength_unit = 'nm'
    
    pyplot.figure()
    pyplot.title(r'$Redistributed~Wavelengths$')
    pyplot.xlabel(r'$Wavelength~(m)$')
    pyplot.ylabel(r'$Intensity~(arbitrary~units)$')
    pyplot.xscale('log')
    pyplot.plot(output_wavelengths, intensity,
                label=f'$\\lambda_0 = {compact_test_wavelength}~{compact_test_wavelength_unit}$')
    pyplot.legend()
    pyplot.show()
