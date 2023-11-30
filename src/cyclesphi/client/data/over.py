import bpy

prefs = bpy.context.preferences.addons['cycles'].preferences
devices = prefs.get_devices() #update

prefs.compute_device_type = 'CLIENT'

devices = prefs.get_devices() #update

for scene in bpy.data.scenes:
    # scene.cycles.device = 'GPU'
    scene.cycles.device = 'CPU'

