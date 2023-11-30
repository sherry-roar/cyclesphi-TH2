/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#pragma once

#include <ostream>  // NOLINT

#include "util/string.h"
#include "util/vector.h"

#include "kernel/types.h"

#ifdef BLENDER_CLIENT
#include "graph/node_enum.h"
#else
#include "graph/node.h"
#endif

CCL_NAMESPACE_BEGIN

const char *pass_type_as_string(const PassType type);

enum class PassMode {
  NOISY,
  DENOISED,
};
const char *pass_mode_as_string(PassMode mode);
std::ostream &operator<<(std::ostream &os, PassMode mode);

struct PassInfo {
  int num_components = -1;
  bool use_filter = false;
  bool use_exposure = false;
  bool is_written = true;
  PassType divide_type = PASS_NONE;
  PassType direct_type = PASS_NONE;
  PassType indirect_type = PASS_NONE;

  /* Pass access for read can not happen directly and needs some sort of compositing (for example,
   * light passes due to divide_type, or shadow catcher pass. */
  bool use_compositing = false;

  /* Used to disable albedo pass for denoising.
   * Light and shadow catcher passes should not have discontinuity in the denoised result based on
   * the underlying albedo. */
  bool use_denoising_albedo = true;

  /* Pass supports denoising. */
  bool support_denoise = false;
};

#ifdef BLENDER_CLIENT
class Pass {
 protected:
  PassType type;
  PassMode mode;
  std::string name;
  bool include_albedo;
  std::string lightgroup;

 public:
  PassType get_type() const
  {
    return type;
  }

  PassMode get_mode() const
  {
    return mode;
  }

  std::string get_name() const
  {
    return name;
  }

  bool get_include_albedo() const
  {
    return include_albedo;
  }

  std::string get_lightgroup() const
  {
    return lightgroup;
  }

  void set_type(PassType t)
  {
    type = t;
  }

  void set_mode(PassMode m)
  {
    mode = m;
  }

  void set_name(std::string n)
  {
    name = n;
  }

  void set_include_albedo(bool a)
  {
    include_albedo = a;
  }

  void set_lightgroup(std::string l)
  {
    lightgroup = l;
  }

#else
class Pass : public Node {
 public:
  NODE_DECLARE

  NODE_SOCKET_API(PassType, type)
  NODE_SOCKET_API(PassMode, mode)
  NODE_SOCKET_API(ustring, name)
  NODE_SOCKET_API(bool, include_albedo)
  NODE_SOCKET_API(ustring, lightgroup)
#endif

  Pass();

  PassInfo get_info() const;

  /* The pass is written by the render pipeline (kernel or denoiser). If the pass is written it
   * will have pixels allocated in a RenderBuffer. Passes which are not written do not have their
   * pixels allocated to save memory. */
  bool is_written() const;

 protected:
  /* The has been created automatically as a requirement to various rendering functionality (such
   * as adaptive sampling). */
  bool is_auto_;

 public:
  static const NodeEnum *get_type_enum();
  static const NodeEnum *get_mode_enum();

  static PassInfo get_info(PassType type,
                           const bool include_albedo = false,
                           const bool is_lightgroup = false);

  static bool contains(const vector<Pass *> &passes, PassType type);

  /* Returns nullptr if there is no pass with the given name or type+mode. */
  static const Pass *find(const vector<Pass *> &passes, const string &name);
  static const Pass *find(const vector<Pass *> &passes,
                          PassType type,
                          PassMode mode = PassMode::NOISY,
#  ifdef BLENDER_CLIENT
                          const string &lightgroup = string()
#else
                          const ustring &lightgroup = ustring()
#endif
                );

  /* Returns PASS_UNUSED if there is no corresponding pass. */
  static int get_offset(const vector<Pass *> &passes, const Pass *pass);

  friend class Film;
};

std::ostream &operator<<(std::ostream &os, const Pass &pass);

CCL_NAMESPACE_END
