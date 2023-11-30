#ifndef __CYCLESPHI_H__
#define __CYCLESPHI_H__

#ifndef RE_API
#  ifdef __cplusplus
#ifdef _WIN32
#    define RE_API __declspec(dllimport)
#else
#    define RE_API
#endif
#  else
#    define RE_API
#  endif
#endif

RE_API void re_init();
RE_API void re_render_frame(char *packet);
RE_API void re_submit_frame(char *packet, char *sbs_image_data);

#endif
